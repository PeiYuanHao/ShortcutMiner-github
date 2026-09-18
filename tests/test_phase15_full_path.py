"""Offline production-path smoke for real model-reliance execution."""

import json
from pathlib import Path

from shortcutminer.evaluation.blind.downstream import downstream_impact_from_output
from shortcutminer.evaluation.blind.evaluator import BlindEvaluator
from shortcutminer.evaluation.blind.generator import BlindChallengeGenerator
from shortcutminer.evaluation.blind.protocol import BlindChallengeConfig
from shortcutminer.evaluation.blind.snapshot import build_evaluation_snapshot
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.inspection.semantic_provider import OpenAICompatibleSemanticVisualInspector
from shortcutminer.protocol.batches import QualitativeResultBatch, QuantitativeResultBatch
from shortcutminer.protocol.evidence import EvidenceLedger
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.run import RunRequest
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.runner import resume_run, run
from shortcutminer.runtime.demo import DemoAgentClient
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.protocol import ToolResult
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime
from test_real_qualitative_workflow import RealEvidenceClient


class Phase15Reasoning:
    def __init__(self):
        self.demo = DemoAgentClient()
        self.qualitative = RealEvidenceClient()

    def invoke_agent(self, agent_name, payload):
        if agent_name == "identify":
            output = self.demo._identify(payload)
            candidate = output["candidates"][0]
            candidate.update({
                "cue": {"family": "spatial", "name": "top-left corner pattern"},
                "description": "A visible corner pattern may co-vary with the class label.",
                "observation": "Public visual descriptors and semantic observations identify a top-left appearance difference.",
                "hypothesis": "The task model may rely on the top-left pattern instead of the central shape.",
                "evidence_sources": ["sample_inspection", "image_statistics"],
            })
            return output
        if agent_name == "executive":
            return {
                "decisions": [{
                    "candidate_id": "C001", "route": "both",
                    "routing_reason": "Deep model-reliance validation and visual inspection are available.",
                    "quantitative": {
                        "objective": "Measure candidate-specific model behavior without reading private truth.",
                        "requested_checks": ["task_model_training", "task_model_evaluation"],
                    },
                    "qualitative": {
                        "objective": "Inspect the visible candidate on bounded samples.",
                        "requested_checks": ["sample_inspection"],
                    },
                    "missing_requirements": [], "limitations": [],
                }],
                "limitations": [],
            }
        if agent_name == "quantitative":
            return self._quantitative(payload)
        if agent_name == "qualitative":
            return self.qualitative.invoke_agent(agent_name, payload)
        return self.demo._summarize(payload)

    def _quantitative(self, payload):
        results = payload["tool_runtime"]["results"]
        if len(results) == 1:
            evaluation = next(
                step for step in payload["experiment_plan_summary"]["steps"]
                if step["experiment_type"] == "task_model_evaluation"
                and step["execution_state"] == "ready"
            )
            return {
                "tool_calls": [{
                    "name": "run_experiment",
                    "arguments": {"plan_step_id": evaluation["step_id"]},
                }]
            }
        return {
            "status": "completed",
            "analysis_type": "task_model_training_and_evaluation",
            "evidence_level": "Q3",
            "evidence_direction": "mixed",
            "evidence_summary": "The frozen reference model was trained and evaluated; transformation-dependent work remains explicitly unmaterialized.",
            "evidence_tool_call_ids": [item["tool_call_id"] for item in results],
            "additional_limitations": ["OOD and counterfactual steps require an explicit deterministic transformation specification."],
            "recommended_followup": ["Define a bounded transformation policy before intervention execution."],
        }


class Phase15Transport:
    def __init__(self):
        self.reasoning = Phase15Reasoning()
        self.calls = []

    def post_json(self, _url, *, headers, body, timeout):
        self.calls.append(body)
        content = body["messages"][-1]["content"]
        if isinstance(content, list):
            public = json.loads(content[0]["text"])
            output = {
                "schema_version": "1.0",
                "observations": [{
                    "sample_id": sample["sample_id"],
                    "sample_reference": sample["sample_reference"],
                    "observation_type": "spatial_pattern", "region": "top-left",
                    "description": "A visible high-contrast corner pattern is present in this public sample.",
                    "confidence": "medium", "limitations": ["Synthetic public-fixture observation."],
                } for sample in public["samples_in_image_order"]],
            }
        else:
            payload = json.loads(content)
            output = self.reasoning.invoke_agent(payload["request"]["target"], payload)
        return {
            "choices": [{"message": {"content": json.dumps(output)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }


def test_full_model_reliance_workflow_reaches_blind_evaluation_and_resumes(tmp_path: Path):
    generated = BlindChallengeGenerator().generate(
        BlindChallengeConfig(
            challenge_id="challenge-juliet", family="spatial_visual",
            challenge_generation_seed=91, sample_count=48, injection_strength="strong",
        ), tmp_path / "challenge",
    )
    runs_root = tmp_path / "runs"
    run_id = "phase15-full-path"
    request = RunRequest(
        run_id=run_id, runs_root=str(runs_root),
        dataset_info={
            "name": "phase15-public-fixture", "root": str(generated.public_dataset_root),
            "metadata_file": "metadata.csv",
        },
        task_input=generated.challenge.public.task, seed=33,
        identify_config={
            "budget": {"max_samples": 8, "max_candidates": 1, "max_dataset_entries": 200, "max_metadata_rows": 100},
            "visual_discovery": {
                "max_samples": 8, "sample_id_column": "id", "sample_path_column": "path",
                "semantic_enabled": True,
            },
        },
        executive_config={
            "audit_options": {
                "validation_depth": "deep", "max_tasks": 2, "max_samples_per_task": 20,
                "allowed_targets": ["quantitative", "qualitative"],
                "runtime_constraints": ["no_external_services"],
            },
            "capabilities": {
                "available_targets": ["quantitative", "qualitative"],
                "available_metadata_fields": ["id", "label", "site", "device", "batch", "measure", "path"],
                "quantitative_checks": ["task_model_training", "task_model_evaluation", "ood_evaluation", "counterfactual_intervention"],
                "qualitative_checks": ["sample_inspection"],
                "model_capabilities": ["retraining", "checkpoint", "intervention_inference"],
                "sample_access": True, "external_services": False,
                "available_output_types": ["metrics", "observations"],
            },
        },
        quantitative_config={
            "execution_resources": {
                "seed": 33,
                "sample_id_column": "id",
                "sample_path_column": "path",
                "max_rows": 100,
                "max_samples": 30,
                "bootstrap_iterations": 20,
            }
        },
        qualitative_config={
            "preparation": {
                "seed": 33, "sampling_strategy": "group_balanced", "sample_id_column": "id",
                "sample_path_column": "path", "group_columns": ["label"],
                "max_evidence_records": 4, "max_metadata_rows": 100,
            }
        },
        metadata={"evaluation_mode": "blind_public_only"},
    )
    model_config = OpenAICompatibleModelConfig(
        model="fake-phase15-multimodal", base_url="https://fake-provider.example/v1",
        api_key_env="SHORTCUTMINER_FAKE_KEY", capabilities={"supports_image_input": True},
        model_call_budget={"max_total_calls": 30, "max_calls_per_agent": 12},
    )
    transport = Phase15Transport()
    client = OpenAICompatibleClient(
        model_config, api_key="phase15-fake-secret", transport=transport, runs_root=runs_root
    )
    registry = build_default_tool_registry(experiment_executor=RealExperimentExecutor())
    runtimes = {
        agent: ToolRuntime(
            registry, run_id=run_id, agent_name=agent,
            dataset_root=generated.public_dataset_root, runs_root=runs_root,
            model_capabilities={"retraining", "checkpoint", "intervention_inference", "predictions", "logits"},
            budget=ToolBudget(max_tool_calls=10, timeout_seconds=30),
        )
        for agent in ("quantitative", "qualitative")
    }
    execution = run(
        request, agent_client=client, tool_runtimes=runtimes,
        semantic_inspector=OpenAICompatibleSemanticVisualInspector(client),
    )
    state = execution.workflow_state
    assert state["identify_context"]["visual_discovery"]["semantic"]["runtime_kind"] == "real_provider"
    quantitative_history = [
        ToolResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((runs_root / run_id / "tools" / "quantitative").glob("*.json"))
    ]
    assert len(quantitative_history) == 2
    assert {item.output["metadata"]["experiment_type"] for item in quantitative_history} == {
        "task_model_training", "task_model_evaluation"
    }
    assert state["qualitative_result"]["results"][0]["payload"]["observations"]
    ledger = EvidenceLedger.model_validate(state["evidence_ledger"])
    assert ledger.candidates[0].records[0].evidence_kind == "quantitative_model_reliance"
    assert ledger.candidates[0].records[0].metrics["model_hash"]

    snapshot = build_evaluation_snapshot(
        repository_root=Path.cwd(), git_commit="b" * 40,
        runtime_config={"seed": 33}, model_config=model_config.model_dump(mode="json"),
        model_runtime_snapshot=client.runtime_snapshot.model_dump(mode="json"),
        challenge_public_hash=generated.challenge.public.public_dataset_hash,
        runtime_sampling_seed=33, evaluation_bootstrap_seed=44, candidate_limit=1,
        validation_depth="deep", budget_settings={"max_intervention_samples": 20},
    )
    evaluated = BlindEvaluator().evaluate(
        challenge=generated.challenge, challenge_root=generated.challenge_root,
        hidden_truth_path=generated.hidden_truth_path, snapshot=snapshot, run_id=run_id,
        candidate_report=CandidateReport.model_validate(state["identify_result"]),
        routing_plan=RoutingPlan.model_validate(state["executive_result"]),
        quantitative_results=QuantitativeResultBatch.model_validate(state["quantitative_result"]),
        qualitative_results=QualitativeResultBatch.model_validate(state["qualitative_result"]),
        evidence_ledger=ledger, summary_report=SummaryReport.model_validate(state["summarize_result"]),
        downstream_impact=downstream_impact_from_output(next(
            item.output for item in quantitative_history
            if item.output["metadata"]["experiment_type"] == "task_model_evaluation"
        )),
        runs_root=runs_root,
    )
    assert evaluated.downstream_impact.available
    assert evaluated.downstream_impact.metric_before is not None
    assert evaluated.downstream_impact.intervention_delta is None
    artifact_mtimes = {
        path: path.stat().st_mtime_ns
        for path in (runs_root / run_id).rglob("*")
        if path.is_file() and (
            "models" in path.parts or "interventions" in path.parts
        )
    }
    resumed = resume_run(
        run_id, runs_root=runs_root, agent_client=client, tool_runtimes=runtimes,
        semantic_inspector=OpenAICompatibleSemanticVisualInspector(client),
    )
    assert resumed.status == "COMPLETED"
    assert runtimes["quantitative"].history == []
    assert {path: path.stat().st_mtime_ns for path in artifact_mtimes} == artifact_mtimes
    assert "phase15-fake-secret" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (runs_root / run_id).rglob("*.json")
    )
