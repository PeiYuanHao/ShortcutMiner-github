"""Offline fake-HTTP smoke through the production model and evidence paths."""

import json
from pathlib import Path

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
from shortcutminer.runner import run
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime
from test_real_qualitative_workflow import RealEvidenceClient


class FullPathTransport:
    def __init__(self):
        self.reasoning = RealEvidenceClient()
        self.calls = []

    def post_json(self, url, *, headers, body, timeout):
        self.calls.append(body)
        content = body["messages"][-1]["content"]
        if isinstance(content, list):
            public = json.loads(content[0]["text"])
            observations = [
                {
                    "sample_id": sample["sample_id"],
                    "sample_reference": sample["sample_reference"],
                    "observation_type": "background_context",
                    "region": "full image",
                    "description": "green-toned visible background",
                    "confidence": "medium",
                    "limitations": ["Synthetic fixture observation."],
                }
                for sample in public["samples_in_image_order"]
            ]
            output = {"schema_version": "1.0", "observations": observations}
        else:
            payload = json.loads(content)
            agent = payload["request"]["target"]
            output = self.reasoning.invoke_agent(agent, payload)
            if agent == "identify":
                candidate = output["candidates"][0]
                candidate["cue"] = {"family": "context", "name": "visible background"}
                candidate["description"] = "Visible background context may co-vary with labels."
                candidate["observation"] = "Semantic observations describe visible backgrounds."
                candidate["hypothesis"] = "Background context may be spuriously associated with label."
                candidate["evidence_sources"] = ["sample_inspection"]
            elif agent == "executive":
                for task in output["tasks"]:
                    task["inputs"]["metadata_fields"] = ["site", "id", "path"]
            elif agent == "quantitative" and "tool_calls" in output:
                output["tool_calls"][0]["arguments"]["parameters"]["cue_field"] = "site"
        return {
            "choices": [{"message": {"content": json.dumps(output)}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }


def test_fake_http_full_production_path_reaches_blind_evaluator(tmp_path: Path):
    generated = BlindChallengeGenerator().generate(
        BlindChallengeConfig(
            challenge_id="challenge-foxtrot",
            family="metadata_association",
            challenge_generation_seed=73,
            sample_count=40,
            injection_strength="strong",
        ),
        tmp_path / "challenge",
    )
    runs_root = tmp_path / "runs"
    run_id = "phase14-full-production-path"
    request = RunRequest(
        run_id=run_id,
        runs_root=str(runs_root),
        dataset_info={
            "name": "neutral-public-challenge",
            "root": str(generated.public_dataset_root),
            "metadata_file": "metadata.csv",
        },
        task_input=generated.challenge.public.task,
        seed=generated.challenge.public.runtime_sampling_seed,
        identify_config={
            "budget": {
                "max_samples": 4,
                "max_candidates": 1,
                "max_dataset_entries": 100,
                "max_metadata_rows": 100,
            },
            "visual_discovery": {
                "max_samples": 4,
                "sample_id_column": "id",
                "sample_path_column": "path",
                "semantic_enabled": True,
            },
        },
        executive_config={
            "audit_options": {
                "validation_depth": "standard",
                "max_tasks": 2,
                "max_samples_per_task": 4,
                "allowed_targets": ["quantitative", "qualitative"],
                "runtime_constraints": ["no_external_services"],
            },
            "capabilities": {
                "available_targets": ["quantitative", "qualitative"],
                "available_metadata_fields": [
                    "id", "label", "site", "device", "batch", "measure", "path"
                ],
                "quantitative_checks": ["association_test"],
                "qualitative_checks": ["sample_inspection"],
                "sample_access": True,
                "external_services": False,
                "available_output_types": ["metrics", "observations"],
            },
        },
        qualitative_config={
            "preparation": {
                "seed": generated.challenge.public.runtime_sampling_seed,
                "sampling_strategy": "group_balanced",
                "sample_id_column": "id",
                "sample_path_column": "path",
                "group_columns": ["label"],
                "max_evidence_records": 4,
                "max_metadata_rows": 100,
            }
        },
        metadata={"evaluation_mode": "blind_public_only"},
    )
    model_config = OpenAICompatibleModelConfig(
        model="fake-multimodal-model",
        base_url="https://fake-provider.example/v1",
        api_key_env="SHORTCUTMINER_FAKE_KEY",
        capabilities={"supports_image_input": True},
        model_call_budget={"max_total_calls": 20, "max_calls_per_agent": 8},
    )
    transport = FullPathTransport()
    client = OpenAICompatibleClient(
        model_config,
        api_key="fake-secret-not-for-storage",
        transport=transport,
        runs_root=runs_root,
    )
    registry = build_default_tool_registry(experiment_executor=RealExperimentExecutor())
    runtimes = {
        agent: ToolRuntime(
            registry,
            run_id=run_id,
            agent_name=agent,
            dataset_root=generated.public_dataset_root,
            runs_root=runs_root,
        )
        for agent in ("quantitative", "qualitative")
    }
    execution = run(
        request,
        agent_client=client,
        tool_runtimes=runtimes,
        semantic_inspector=OpenAICompatibleSemanticVisualInspector(client),
    )
    state = execution.workflow_state
    assert state["identify_context"]["visual_discovery"]["semantic"]["runtime_kind"] == "real_provider"
    assert state["identify_result"]["candidates"][0]["cue"]["name"] == "visible background"
    assert any(
        reference.startswith("experiments/")
        for reference in state["quantitative_result"]["results"][0]["payload"]["artifact_references"]
    )
    assert state["qualitative_result"]["results"][0]["payload"]["observations"]
    assert state["evidence_ledger"]["fingerprint"]

    snapshot = build_evaluation_snapshot(
        repository_root=Path.cwd(),
        git_commit="a" * 40,
        runtime_config={"seed": request.seed},
        model_config=model_config.model_dump(mode="json"),
        model_runtime_snapshot=client.runtime_snapshot.model_dump(mode="json"),
        challenge_public_hash=generated.challenge.public.public_dataset_hash,
        runtime_sampling_seed=request.seed,
        evaluation_bootstrap_seed=19,
        candidate_limit=1,
        validation_depth="standard",
        budget_settings=model_config.model_call_budget.model_dump(mode="json"),
    )
    evaluated = BlindEvaluator().evaluate(
        challenge=generated.challenge,
        challenge_root=generated.challenge_root,
        hidden_truth_path=generated.hidden_truth_path,
        snapshot=snapshot,
        run_id=run_id,
        candidate_report=CandidateReport.model_validate(state["identify_result"]),
        routing_plan=RoutingPlan.model_validate(state["executive_result"]),
        quantitative_results=QuantitativeResultBatch.model_validate(state["quantitative_result"]),
        qualitative_results=QualitativeResultBatch.model_validate(state["qualitative_result"]),
        evidence_ledger=EvidenceLedger.model_validate(state["evidence_ledger"]),
        summary_report=SummaryReport.model_validate(state["summarize_result"]),
        runs_root=runs_root,
    )
    assert evaluated.run_id == run_id
    assert (runs_root / run_id / "evaluation" / "blind_result.json").is_file()
    assert len(client.invocations) >= 7
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (runs_root / run_id).rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )
    assert "fake-secret-not-for-storage" not in persisted
    assert "base64," not in persisted
