from pathlib import Path

from shortcutminer.experiment.qualitative_context import (
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
        (root / "images" / name).write_bytes(b"\x89PNG\r\n\x1a\nmock")


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


def test_missing_metadata_produces_degraded_context(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    context = build_qualitative_context(_request(tmp_path, metadata_file="missing.csv"))

    assert context.metadata_available is False
    assert context.samples_selected == 0
    assert any("unavailable or invalid" in item for item in context.limitations)


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
