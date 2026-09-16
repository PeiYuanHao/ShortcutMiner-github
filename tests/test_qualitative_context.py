from pathlib import Path

from PIL import Image
import pytest

from shortcutminer.experiment.qualitative_context import (
    QualitativeContextError,
    build_qualitative_context,
    build_qualitative_request,
)
from shortcutminer.protocol.executive import AnalysisTask, AnalysisTaskInputs
from shortcutminer.protocol.identify import Candidate, Cue
from shortcutminer.protocol.qualitative import (
    QualitativePreparationOptions,
    QualitativeRequest,
)
from shortcutminer.protocol.executive import RoutingPlan
from test_executive_executor import (
    _request as _executive_request,
    _valid_output as _executive_output,
)


def _request(dataset_root: Path, *, metadata_file: str = "metadata.csv") -> QualitativeRequest:
    candidate = Candidate(
        candidate_id="C001",
        cue=Cue(family="appearance", name="dark_border"),
        description="A border may differ by label.",
        observation="Metadata suggests a candidate.",
        hypothesis="A dark border may be associated with the label.",
        evidence_sources=["metadata"],
        priority="high",
        status="unverified",
    )
    task = AnalysisTask(
        protocol_version="1.0",
        run_id="qualitative_test_run",
        message_id="executive:task:C001:QL",
        source="executive",
        target="qualitative",
        message_type="analysis_task",
        in_reply_to="executive:plan:001",
        task_id="TASK-C001-QL",
        candidate_id="C001",
        hypothesis=candidate.hypothesis,
        objective="Inspect border appearance across labels.",
        inputs=AnalysisTaskInputs(
            dataset="test_dataset",
            dataset_reference=str(dataset_root),
            label_column="label",
            split="train",
            metadata_fields=["sample_id", "filename", "label", "site"],
            artifact_references=[],
            sample_limit=4,
            requires_sample_access=True,
            requires_external_services=False,
            runtime_constraints=["no_retraining"],
            required_model_capabilities=[],
        ),
        requested_checks=["sample_inspection"],
        expected_outputs=["observations"],
        priority="high",
        dependencies=[],
        status="planned",
    )
    return QualitativeRequest(
        protocol_version="1.0",
        run_id="qualitative_test_run",
        message_id="qualitative:request:C001",
        source="orchestrator",
        target="qualitative",
        message_type="qualitative_analysis_request",
        in_reply_to=task.message_id,
        candidate=candidate,
        task=task,
        metadata_file=metadata_file,
        preparation=QualitativePreparationOptions(
            sampling_strategy="group_balanced",
            seed=42,
            sample_id_column="sample_id",
            sample_path_column="filename",
            split_column="split",
            group_columns=["label"],
            max_evidence_records=4,
            max_metadata_rows=100,
        ),
        limitations=["Upstream evidence was limited."],
    )


def _dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.csv").write_text(
        "sample_id,filename,label,site,split\n"
        "S1,images/s1.png,A,X,train\n"
        "S2,images/s2.png,A,Y,train\n"
        "S3,images/s3.png,B,X,train\n"
        "S4,images/s4.png,B,Y,train\n",
        encoding="utf-8",
    )
    (root / "images").mkdir()
    for name in ("s1.png", "s2.png", "s3.png", "s4.png"):
        Image.new("RGB", (16, 16), color=(64, 96, 128)).save(
            root / "images" / name
        )


def test_context_builds_balanced_traceable_manifest(tmp_path: Path):
    _dataset(tmp_path)
    context = build_qualitative_context(_request(tmp_path))

    assert context.samples_selected == 4
    assert context.sample_files_available == 4
    assert {sample.sample_id for sample in context.selected_samples} == {
        "S1", "S2", "S3", "S4"
    }
    assert context.group_coverage["selected::label=A"] == 2
    assert context.group_coverage["selected::label=B"] == 2
    assert context.selected_samples[0].sample_reference.startswith("images/")
    assert "Upstream evidence was limited." in context.limitations


def test_label_values_are_exposed_even_when_not_repeated_in_metadata_fields(
    tmp_path: Path,
):
    _dataset(tmp_path)
    request = _request(tmp_path)
    request.task.inputs.metadata_fields.remove("label")

    context = build_qualitative_context(request)

    assert "label" in context.metadata_fields
    assert all("label" in sample.metadata for sample in context.selected_samples)


def test_missing_metadata_produces_degraded_context(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    context = build_qualitative_context(_request(tmp_path, metadata_file="missing.csv"))

    assert context.metadata_available is False
    assert context.samples_selected == 0
    assert any("unavailable or invalid" in item for item in context.limitations)


def test_requested_split_without_configured_column_is_degraded(tmp_path: Path):
    _dataset(tmp_path)
    request = _request(tmp_path)
    request.preparation.split_column = None

    context = build_qualitative_context(request)

    assert context.samples_selected == 0
    assert any("authorized split column" in item for item in context.limitations)


def test_sample_path_escape_is_not_exposed(tmp_path: Path):
    _dataset(tmp_path)
    metadata = tmp_path / "metadata.csv"
    content = metadata.read_text("utf-8").replace("images/s1.png", "../secret.png")
    metadata.write_text(content, encoding="utf-8")

    context = build_qualitative_context(_request(tmp_path))
    sample = next(item for item in context.selected_samples if item.sample_id == "S1")
    assert sample.sample_reference is None
    assert sample.sample_available is False
    assert any("path was rejected" in item for item in context.limitations)


def test_declared_image_format_is_verified(tmp_path: Path):
    _dataset(tmp_path)
    (tmp_path / "images" / "s1.png").write_bytes(b"not-a-png")

    context = build_qualitative_context(_request(tmp_path))
    sample = next(item for item in context.selected_samples if item.sample_id == "S1")
    assert sample.sample_available is False
    assert any("declared image format" in item for item in context.limitations)


def test_request_is_built_from_validated_executive_state(tmp_path: Path):
    executive_request = _executive_request(tmp_path)
    routing_plan = RoutingPlan.model_validate(_executive_output(executive_request))
    request = build_qualitative_request(
        executive_request=executive_request,
        routing_plan=routing_plan,
        task_id="TASK-C001-QL",
        message_id="qualitative:request:C001",
        preparation=QualitativePreparationOptions(
            sampling_strategy="group_balanced",
            seed=42,
            group_columns=["site"],
            max_evidence_records=20,
        ),
    )

    assert request.candidate == executive_request.candidate_report.candidates[0]
    assert request.task == routing_plan.tasks[1]
    assert request.in_reply_to == routing_plan.tasks[1].message_id


def _addressing_state(tmp_path: Path):
    executive_request = _executive_request(tmp_path)
    executive_request.routing_context.capabilities.available_metadata_fields = [
        "id", "label", "path", "device", "site",
    ]
    output = _executive_output(executive_request)
    qualitative_task = output["tasks"][1]
    qualitative_task["inputs"]["metadata_fields"] = []
    routing_plan = RoutingPlan.model_validate(output)
    return executive_request, routing_plan


def _addressing_preparation(**overrides) -> QualitativePreparationOptions:
    values = {
        "sampling_strategy": "group_balanced",
        "seed": 42,
        "sample_id_column": "id",
        "sample_path_column": "path",
        "group_columns": ["label"],
        "max_evidence_records": 20,
    }
    values.update(overrides)
    return QualitativePreparationOptions.model_validate(values)


def test_addressing_columns_use_authoritative_dataset_fields(tmp_path: Path):
    executive_request, routing_plan = _addressing_state(tmp_path)

    request = build_qualitative_request(
        executive_request=executive_request,
        routing_plan=routing_plan,
        task_id="TASK-C001-QL",
        message_id="qualitative:request:C001",
        preparation=_addressing_preparation(),
    )

    assert request.preparation.sample_id_column == "id"
    assert request.preparation.sample_path_column == "path"
    assert request.task.inputs.metadata_fields == []


@pytest.mark.parametrize(
    ("override", "unknown"),
    [
        ({"sample_id_column": "unknown_id"}, "unknown_id"),
        ({"sample_path_column": "unknown_path"}, "unknown_path"),
    ],
)
def test_unknown_addressing_column_is_rejected(
    tmp_path: Path, override: dict, unknown: str,
):
    executive_request, routing_plan = _addressing_state(tmp_path)

    with pytest.raises(QualitativeContextError, match=unknown):
        build_qualitative_request(
            executive_request=executive_request,
            routing_plan=routing_plan,
            task_id="TASK-C001-QL",
            message_id="qualitative:request:C001",
            preparation=_addressing_preparation(**override),
        )


def test_group_column_requires_task_evidence_authorization(tmp_path: Path):
    executive_request, routing_plan = _addressing_state(tmp_path)

    with pytest.raises(QualitativeContextError, match="grouping"):
        build_qualitative_request(
            executive_request=executive_request,
            routing_plan=routing_plan,
            task_id="TASK-C001-QL",
            message_id="qualitative:request:C001",
            preparation=_addressing_preparation(group_columns=["device"]),
        )

    output = routing_plan.model_dump(mode="json")
    output["tasks"][1]["inputs"]["metadata_fields"] = ["device"]
    authorized_plan = RoutingPlan.model_validate(output)
    request = build_qualitative_request(
        executive_request=executive_request,
        routing_plan=authorized_plan,
        task_id="TASK-C001-QL",
        message_id="qualitative:request:C001",
        preparation=_addressing_preparation(group_columns=["device"]),
    )
    assert request.preparation.group_columns == ["device"]
