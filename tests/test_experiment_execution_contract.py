from pathlib import Path
from typing import get_args

import pandas as pd
import pytest
from PIL import Image

from shortcutminer.experiment.execution_contract import (
    EXPERIMENT_PARAMETER_OWNERSHIP,
    AuthoritativeResourceMissing,
    ExperimentDependencyPending,
    UnauthorizedScientificField,
    UnsafeResourcePath,
    _normalize_authorized_path,
    classify_experiment_availability,
)
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.experiment.tool_executor import ExperimentType
from shortcutminer.protocol.quantitative import (
    QuantitativeExecutionRequirement,
    QuantitativeExecutionResources,
)
from shortcutminer.protocol.experiment import (
    MetadataAblationParameters,
    MetadataInterventionParameters,
    MetricComputationParameters,
    OODEvaluationParameters,
    PredictiveProbeParameters,
    StatisticalCheckParameters,
    SubgroupEvaluationParameters,
    TaskModelEvaluationParameters,
    TaskModelTrainingParameters,
    CounterfactualInterventionParameters,
)
from shortcutminer.tooling.agent_loop import finalize_tool_call_draft
from shortcutminer.tooling.builtins import ExperimentToolInput, build_default_tool_registry
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime


ALL_TYPES = list(get_args(ExperimentType))
PARAMETER_MODELS = {
    "statistical_check": StatisticalCheckParameters,
    "subgroup_evaluation": SubgroupEvaluationParameters,
    "model_evaluation": PredictiveProbeParameters,
    "intervention": MetadataInterventionParameters,
    "ablation": MetadataAblationParameters,
    "metric_computation": MetricComputationParameters,
    "task_model_training": TaskModelTrainingParameters,
    "task_model_evaluation": TaskModelEvaluationParameters,
    "ood_evaluation": OODEvaluationParameters,
    "counterfactual_intervention": CounterfactualInterventionParameters,
}


def _fixture(tmp_path: Path):
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    rows = []
    for index in range(40):
        label = index % 2
        reference = f"images/{index:03d}.png"
        Image.new(
            "RGB", (16, 16), (240, 40, 40) if label else (40, 40, 240)
        ).save(dataset / reference)
        rows.append({
            "id": f"S{index:03d}",
            "path": reference,
            "label": label,
            "cue": "red" if label else "blue",
            "group": "g1" if index < 20 else "g2",
            "noise": index % 4,
            "prediction": label,
            "split": "train" if index < 30 else "test",
        })
    pd.DataFrame(rows).to_csv(dataset / "metadata.csv", index=False)
    runs = tmp_path / "runs"
    (runs / "contract-run").mkdir(parents=True)
    runtime = ToolRuntime(
        build_default_tool_registry(experiment_executor=RealExperimentExecutor()),
        run_id="contract-run",
        agent_name="quantitative",
        dataset_root=dataset,
        runs_root=runs,
        model_capabilities={
            "predictions", "logits", "checkpoint", "retraining",
            "intervention_inference",
        },
        budget=ToolBudget(max_tool_calls=20, timeout_seconds=30),
    )
    resources = QuantitativeExecutionResources(
        dataset_name="contract_dataset",
        metadata_file="metadata.csv",
        label_column="label",
        available_metadata_fields=[
            "id", "path", "label", "cue", "group", "noise", "prediction", "split"
        ],
        allowed_scientific_metadata_fields=["cue", "group", "noise", "prediction"],
        sample_id_column="id",
        sample_path_column="path",
        split_column="split",
        available_splits=["train", "test"],
        seed=16001,
        max_rows=40,
        max_samples=40,
        max_permutations=5,
        bootstrap_iterations=10,
        max_model_iterations=100,
        max_intervention_samples=20,
    )
    requirement = QuantitativeExecutionRequirement(
        must_attempt_run_experiment=True,
        reason="task_is_executable",
        resolved_experiment_types=ALL_TYPES,
        available_now_experiment_types=[
            item for item in ALL_TYPES
            if item not in {"task_model_evaluation", "ood_evaluation", "counterfactual_intervention"}
        ],
        pending_dependency_experiment_types=[
            "task_model_evaluation", "ood_evaluation", "counterfactual_intervention"
        ],
    )
    payload = {
        "request": {
            "run_id": "contract-run",
            "candidate": {"candidate_id": "C001"},
            "task": {"task_id": "TASK-C001-QT"},
        },
        "execution_resources": resources.model_dump(mode="json"),
        "execution_requirement": requirement.model_dump(mode="json"),
    }
    return runtime, payload, dataset, runs / "contract-run"


def _drafts():
    transform = {"type": "brightness", "factor": 0.8}
    return [
        ("statistical_check", {"cue_field": "cue", "n_permutations": 2}),
        ("subgroup_evaluation", {"group_field": "group"}),
        ("model_evaluation", {"feature_fields": ["cue"]}),
        ("intervention", {"cue_field": "cue", "intervention_type": "shuffle"}),
        ("ablation", {"feature_fields": ["cue", "noise"], "ablate_field": "cue"}),
        ("metric_computation", {
            "predictions_file": "metadata.csv",
            "predictions_source": "dataset",
            "prediction_column": "prediction",
            "metric": "accuracy",
        }),
        ("task_model_training", {"class_weight": "none", "holdout_fraction": 0.25}),
        ("task_model_evaluation", {}),
        ("ood_evaluation", {
            "transformation": transform,
            "target_cue": "global brightness",
            "comparison_mode": "paired",
        }),
        ("counterfactual_intervention", {
            "transformation": transform,
            "target_cue": "global brightness",
        }),
    ]


def test_ownership_matrix_is_complete_and_disjoint():
    assert set(EXPERIMENT_PARAMETER_OWNERSHIP) == set(ALL_TYPES)
    for experiment_type, rules in EXPERIMENT_PARAMETER_OWNERSHIP.items():
        fields = [field for values in rules.values() for field in values]
        assert len(fields) == len(set(fields))
        assert set(fields) == set(PARAMETER_MODELS[experiment_type].model_fields)


def test_all_ten_provider_drafts_finalize_and_cross_real_backend_boundary(tmp_path: Path):
    runtime, payload, _dataset, _run_root = _fixture(tmp_path)
    statuses = {}
    for call_index, (experiment_type, parameters) in enumerate(_drafts()):
        call = finalize_tool_call_draft(
            {
                "name": "run_experiment",
                "arguments": {
                    "experiment_type": experiment_type,
                    "parameters": parameters,
                },
            },
            agent_name="quantitative",
            base_payload=payload,
            tool_runtime=runtime,
            iteration=call_index + 1,
            call_index=0,
        )
        canonical = ExperimentToolInput.model_validate(call.arguments)
        assert canonical.experiment_type == experiment_type
        result = runtime.execute(call)
        statuses[experiment_type] = result.status
        assert result.status == "success", result.error
    assert set(statuses) == set(ALL_TYPES)


def test_dependency_plan_distinguishes_pending_from_schema_invalid(tmp_path: Path):
    runtime, payload, _dataset, _run_root = _fixture(tmp_path)
    available, pending, unsupported = classify_experiment_availability(
        payload["execution_requirement"]["resolved_experiment_types"],
        history=runtime.history,
        explicit_artifacts=set(),
        run_id=runtime.run_id,
        candidate_id="C001",
        task_id="TASK-C001-QT",
    )
    assert "task_model_training" in available
    assert pending == [
        "task_model_evaluation", "ood_evaluation", "counterfactual_intervention"
    ]
    assert unsupported == []
    with pytest.raises(ExperimentDependencyPending):
        finalize_tool_call_draft(
            {
                "name": "run_experiment",
                "arguments": {
                    "experiment_type": "task_model_evaluation",
                    "parameters": {},
                },
            },
            agent_name="quantitative",
            base_payload=payload,
            tool_runtime=runtime,
            iteration=1,
            call_index=0,
        )


def test_path_policy_matrix(tmp_path: Path):
    runtime, _payload, dataset, run_root = _fixture(tmp_path)
    inside = dataset / "metadata.csv"
    assert _normalize_authorized_path(
        "metadata.csv", root=dataset, authorized=None
    ) == "metadata.csv"
    assert _normalize_authorized_path(
        str(inside), root=dataset, authorized=None
    ) == "metadata.csv"
    with pytest.raises(UnsafeResourcePath):
        _normalize_authorized_path("../outside.csv", root=dataset, authorized=None)
    with pytest.raises(UnsafeResourcePath):
        _normalize_authorized_path("C:\\outside\\file.csv", root=dataset, authorized=None)
    with pytest.raises(UnsafeResourcePath):
        _normalize_authorized_path("/etc/passwd", root=dataset, authorized=None)
    with pytest.raises(AuthoritativeResourceMissing):
        _normalize_authorized_path("missing.csv", root=dataset, authorized=None)
    artifact = run_root / "models" / "model.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(UnsafeResourcePath):
        _normalize_authorized_path(
            "models/model.json", root=run_root, authorized={"other.json"}
        )
    assert not runtime.history


@pytest.mark.parametrize("field", ["id", "path", "label", "unknown"])
def test_column_authority_rejects_addressing_label_and_unknown(tmp_path: Path, field: str):
    runtime, payload, _dataset, _run_root = _fixture(tmp_path)
    with pytest.raises(UnauthorizedScientificField):
        finalize_tool_call_draft(
            {
                "name": "run_experiment",
                "arguments": {
                    "experiment_type": "statistical_check",
                    "parameters": {"cue_field": field},
                },
            },
            agent_name="quantitative",
            base_payload=payload,
            tool_runtime=runtime,
            iteration=1,
            call_index=0,
        )


def test_split_and_feature_authority_are_enforced(tmp_path: Path):
    runtime, payload, _dataset, _run_root = _fixture(tmp_path)
    with pytest.raises(UnauthorizedScientificField):
        finalize_tool_call_draft(
            {
                "name": "run_experiment",
                "arguments": {
                    "experiment_type": "model_evaluation",
                    "parameters": {
                        "feature_fields": ["cue"],
                        "train_split_values": ["train"],
                        "test_split_values": ["unknown"],
                    },
                },
            },
            agent_name="quantitative",
            base_payload=payload,
            tool_runtime=runtime,
            iteration=1,
            call_index=0,
        )
