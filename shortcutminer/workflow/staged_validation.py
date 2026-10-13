"""Controllers around the existing executors, not another workflow or Agent."""
import json

from shortcutminer.experiment.evidence_promotion import (
    make_stage_plan, freeze_stage_plan, stage1_projection, load_or_evaluate_promotion,
    stable_hash, _cache_path, CHEAP_BACKENDS,
)
from shortcutminer.protocol.promotion import PromotionPolicy, ValidationStagePlan, PromotionReport
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.workflow.observability import emit, lifecycle


def audit(state):
    return state.get("metadata", {}).get("entrypoint") == "audit"


def root(state):
    return state["identify_config"]["runs_root"]


def stage_plan_node(state):
    if not audit(state):
        return {}
    policy = PromotionPolicy.model_validate(state.get("metadata", {}).get("promotion_policy", {}))
    probe_candidates = []
    if policy.strategy == 'budgeted_paired_probe':
        seen_quantitative = set()
        for task in state['executive_result']['tasks']:
            if task['target'] == 'quantitative':
                if task['candidate_id'] in seen_quantitative:
                    raise ValueError('Adaptive allocation permits one quantitative probe task per candidate')
                seen_quantitative.add(task['candidate_id'])
            steps = state.get('quantitative_experiment_plans', {}).get(task['task_id'], {}).get('steps', [])
            resources = state.get('quantitative_config', {}).get('execution_resources', {})
            memory_policy = resources.get('ood_recipe_selection') or {}
            memory_fields = set(memory_policy.get('fields', []))
            waiting = [s for s in steps if s['execution_state'] == 'needs_scientific_spec'
                       and memory_fields.intersection(s.get('scientific_scope', []))]
            if (resources.get('adaptive_screening') and any(s['experiment_type'] == 'task_model_training' for s in waiting)
                    and any(s['experiment_type'] in {'ood_evaluation', 'counterfactual_intervention'} for s in waiting)):
                if resources['adaptive_screening']['candidate_budget'] != policy.max_probe_candidates:
                    raise ValueError('Adaptive alpha allocation and promotion candidate budgets differ')
                probe_candidates.append(task['candidate_id'])
            training = [s for s in steps if s['experiment_type'] == 'task_model_training' and s['execution_state'] == 'ready']
            paired = [s for s in steps if s['experiment_type'] in {'ood_evaluation', 'counterfactual_intervention'} and s['execution_state'] in {'ready', 'pending_dependency'}]
            if training and paired and all(s['canonical_input']['parameters'].get('adaptive_screening') for s in training):
                for step in training:
                    if step['canonical_input']['parameters']['adaptive_screening']['candidate_budget'] != policy.max_probe_candidates:
                        raise ValueError('Adaptive alpha allocation and promotion candidate budgets differ')
                probe_candidates.append(task['candidate_id'])
    plan = freeze_stage_plan(root(state), make_stage_plan(state["run_id"], state["executive_result"],
        state.get("quantitative_experiment_plans", {}), policy, probe_candidate_ids=sorted(set(probe_candidates))))
    emit(root(state), state["run_id"], "STAGE_STARTED", "stage1_validation")
    return {"validation_stage_plan": plan.model_dump(mode="json")}


def stage1_node(state, *, function, **arguments):
    if not audit(state):
        return function(state, **arguments)
    output = function(state, validation_phase="stage1", **arguments)
    branch = "quantitative" if "quantitative_result" in output else "qualitative"
    return {**output, f"stage1_{branch}_result": output[f"{branch}_result"]}


def promotion_node(state):
    if not audit(state):
        return {}
    plan = ValidationStagePlan.model_validate(state["validation_stage_plan"])
    emit(root(state), state["run_id"], "STAGE_COMPLETED", "stage1_validation")
    candidates = [item["candidate"]["candidate_id"] for item in state["executive_result"]["decisions"]]
    inputs = stage1_projection(candidates, state["stage1_quantitative_result"], state["stage1_qualitative_result"], run_id=state["run_id"])
    result = load_or_evaluate_promotion(root(state), state["run_id"], plan.policy, inputs, probe_candidate_ids=plan.probe_eligible_candidate_ids)
    emit(root(state), state["run_id"], "PROMOTION_COMPLETED", "evidence_promotion",
         results=[item.model_dump(mode="json") for item in result.results])
    return {"promotion_result": result.model_dump(mode="json")}


def stage2_node(state, *, quantitative_client, qualitative_client, quantitative_runtime=None, qualitative_runtime=None):
    if not audit(state):
        return {}
    from shortcutminer.workflow.nodes import quantitative_node, qualitative_node
    promotion = PromotionReport.model_validate(state["promotion_result"])
    plan = ValidationStagePlan.model_validate(state["validation_stage_plan"])
    promoted = {item.candidate_id for item in promotion.results if item.decision == "promoted"}
    eligible = [task for task in state["executive_result"]["tasks"] if task["candidate_id"] in promoted
                and plan.task_checks[task["task_id"]]["expensive_validation"]]
    output, executions = {}, []
    with lifecycle(root(state), state["run_id"], "stage2_validation", skipped=not eligible):
        for branch, function, client, runtime in (("quantitative", quantitative_node, quantitative_client, quantitative_runtime),
                                                  ("qualitative", qualitative_node, qualitative_client, qualitative_runtime)):
            tasks = [task for task in eligible if task["target"] == branch]
            if not tasks:
                continue
            result = function(state, client=client, tool_runtime=runtime, validation_phase="stage2", selected_task_ids={task["task_id"] for task in tasks})
            replacements = {item["payload"]["task_id"]: item for item in result[f"{branch}_result"]["results"]}
            batch = dict(state[f"stage1_{branch}_result"])
            batch["results"] = [replacements.get(item["payload"]["task_id"], item) for item in batch["results"]]
            output[f"{branch}_result"] = batch
            executions.extend({"candidate_id": task["candidate_id"], "task_id": task["task_id"],
                               "requested_checks": plan.task_checks[task["task_id"]]["expensive_validation"],
                               "status": replacements[task["task_id"]]["payload"].get("status", replacements[task["task_id"]]["payload"].get("evidence_status"))} for task in tasks)
    output["stage2_validation"] = {"status": "completed" if eligible else "skipped", "executions": executions}
    decisions = {item.candidate_id: item.decision for item in promotion.results}
    for branch in ("quantitative", "qualitative"):
        batch = output.get(f"{branch}_result", state[f"stage1_{branch}_result"])
        finalized = []
        for item in batch["results"]:
            payload = item["payload"]
            decision = decisions[payload["candidate_id"]]
            if decision != "promoted" and plan.task_checks[payload["task_id"]]["expensive_validation"]:
                item = {**item, "payload": {**payload, "limitations": list(dict.fromkeys([
                    *payload["limitations"], f"expensive_validation_not_executed:{decision}"]))}}
                if branch == "quantitative":
                    from shortcutminer.storage.stage_artifacts import save_stage_json
                    save_stage_json(root(state), run_id=state["run_id"], stage=branch, scope=payload["task_id"], filename="result.json", payload=item)
                else:
                    from shortcutminer.storage.qualitative_artifacts import save_qualitative_result
                    from shortcutminer.protocol.qualitative import QualitativeResult
                    save_qualitative_result(root(state), QualitativeResult.model_validate(item), scope=payload["task_id"])
            finalized.append(item)
        output[f"{branch}_result"] = {**batch, "results": finalized}
    return output


def guard_runtime(runtime, *, phase, step_ids=None):
    if runtime is None:
        return
    execute = runtime.execute
    def gated_execute(call):
        if phase == "stage1" and call.tool_name == "run_experiment":
            if call.arguments.get("experiment_type") not in CHEAP_BACKENDS or call.experiment_type != call.arguments.get("experiment_type") or call.plan_step_id not in (step_ids or set()):
                raise ValueError("Stage-2 ToolCall forbidden before evidence promotion")
        return execute(call)  # Same budget, retry, permissions and task isolation.
    runtime.execute = gated_execute


def authorize_stage2(state, candidate_id):
    plan = ValidationStagePlan.model_validate(state["validation_stage_plan"])
    path = _cache_path(root(state), state["run_id"], "promotion/result.json")
    if not path.is_file():
        raise ValueError("Stage-2 requires a persisted promotion decision")
    # The gate cache verifies its source/policy/result hashes without invoking work.
    candidates = [item["candidate"]["candidate_id"] for item in state["executive_result"]["decisions"]]
    inputs = stage1_projection(candidates, state["stage1_quantitative_result"], state["stage1_qualitative_result"], run_id=state["run_id"])
    result = load_or_evaluate_promotion(root(state), state["run_id"], plan.policy, inputs, probe_candidate_ids=plan.probe_eligible_candidate_ids)
    if result.model_dump(mode="json") != state["promotion_result"] or not any(item.candidate_id == candidate_id and item.decision == "promoted" for item in result.results):
        raise ValueError("Stage-2 candidate was not evidence-promoted")


def cached_stage1(state, task, branch, function, model):
    path = _cache_path(root(state), state["run_id"], f"validation/stage1/{task.task_id}-{branch}.json")
    identity = stable_hash({"stage_plan_hash": state["validation_stage_plan"]["result_hash"], "task": task.model_dump(mode="json")})
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        result = model.model_validate(value["result"])
        if value["input_hash"] != identity or value["result_hash"] != stable_hash(value["result"]) or result.run_id != state["run_id"] or result.payload.task_id != task.task_id or result.payload.candidate_id != task.candidate_id:
            raise ValueError("Stage-1 cached authority changed")
        return result
    result = function()
    serialized = result.model_dump(mode="json")
    atomic_write_json(path, {"input_hash": identity, "result_hash": stable_hash(serialized), "result": serialized}, error_type=ValueError)
    return result
