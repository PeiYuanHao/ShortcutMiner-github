"""Scientific, safety, contract, and persistence tests for visual discovery."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from shortcutminer.experiment.dataset_profiler import build_dataset_profile
from shortcutminer.experiment.identify_context import build_identify_context_from_profile
from shortcutminer.experiment.identify_executor import (
    IdentifyExecutionError,
    validate_candidate_report,
)
from shortcutminer.experiment.visual_discovery import (
    VisualDiscoveryError,
    VisualDiscoveryProfiler,
)
from shortcutminer.inspection.image import (
    DESCRIPTOR_DEFINITION_VERSION,
    flatten_image_descriptors,
    inspect_image_file,
)
from shortcutminer.inspection.semantic import MockSemanticVisualInspector
from shortcutminer.protocol.identify import (
    CandidateReport,
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)
from shortcutminer.protocol.visual_discovery import (
    SemanticObservation,
    VisualDiscoveryOptions,
    VisualDiscoveryProfile,
)


def _request(
    root: Path,
    *,
    run_id: str = "visual-discovery-run",
    metadata_file: str | None = "metadata.csv",
    max_samples: int = 8,
) -> IdentifyRequest:
    return IdentifyRequest(
        protocol_version="1.0",
        run_id=run_id,
        message_id=f"{run_id}:identify:request",
        source="orchestrator",
        target="identify",
        message_type="identify_request",
        dataset=DatasetSpec(
            name="generic_visual_dataset",
            root=str(root),
            metadata_file=metadata_file,
        ),
        task=TaskSpec(type="classification", label_column="label"),
        budget=IdentifyBudget(
            max_samples=max_samples,
            max_candidates=3,
            max_dataset_entries=1_000,
            max_metadata_rows=1_000,
        ),
    )


def _profile(
    root: Path,
    *,
    run_id: str = "visual-discovery-run",
    metadata_file: str | None = "metadata.csv",
    max_samples: int = 8,
    seed: int = 17,
    options: VisualDiscoveryOptions | None = None,
    runs_root: Path | None = None,
    profiler: VisualDiscoveryProfiler | None = None,
    semantic_inspector=None,
) -> VisualDiscoveryProfile:
    request = _request(
        root,
        run_id=run_id,
        metadata_file=metadata_file,
        max_samples=max_samples,
    )
    dataset_profile = build_dataset_profile(request, seed=seed)
    result = (profiler or VisualDiscoveryProfiler()).profile(
        request,
        dataset_profile,
        seed=seed,
        options=options
        or VisualDiscoveryOptions(
            max_samples=max_samples,
            sample_id_column="sample_id" if metadata_file else None,
            sample_path_column="sample_path" if metadata_file else None,
        ),
        runs_root=runs_root,
        semantic_inspector=semantic_inspector,
    )
    assert result is not None
    return result


def _save_rgb(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def _write_metadata(root: Path, rows: list[tuple[str, int]]) -> None:
    body = "sample_id,sample_path,label\n" + "".join(
        f"{sample_id},{sample_id}.png,{label}\n" for sample_id, label in rows
    )
    (root / "metadata.csv").write_text(body, encoding="utf-8")


def test_canonical_descriptor_fixture_correctness(tmp_path: Path):
    dark = np.full((20, 20, 3), 8, dtype=np.uint8)
    bright = np.full((20, 20, 3), 240, dtype=np.uint8)
    center = np.zeros((20, 20, 3), dtype=np.uint8)
    center[5:15, 5:15] = 255
    corner = np.zeros((20, 20, 3), dtype=np.uint8)
    corner[:4, :4] = 255
    noise = np.random.default_rng(123).integers(
        0, 256, (20, 20, 3), dtype=np.uint8
    )
    red = np.zeros((20, 20, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    fixtures = {
        "dark.png": dark,
        "bright.png": bright,
        "center.png": center,
        "corner.png": corner,
        "noise.png": noise,
        "red.png": red,
    }
    for name, pixels in fixtures.items():
        _save_rgb(tmp_path / name, pixels)

    def inspect(name: str):
        return inspect_image_file(
            tmp_path / name,
            requested_descriptors=list(
                VisualDiscoveryOptions().requested_descriptors
            ),
            max_file_bytes=1_000_000,
            max_pixels=10_000,
            max_frames=1,
        ).descriptors

    outputs = {name: inspect(name) for name in fixtures}
    assert outputs["dark.png"]["intensity"]["mean"] < 0.05
    assert outputs["bright.png"]["intensity"]["mean"] > 0.9
    assert (
        outputs["center.png"]["spatial_stats"]["center"]["mean_intensity"]
        > outputs["center.png"]["spatial_stats"]["border"]["mean_intensity"]
    )
    assert (
        outputs["corner.png"]["spatial_stats"]["top_left"]["mean_intensity"]
        > outputs["corner.png"]["spatial_stats"]["bottom_right"]["mean_intensity"]
    )
    assert outputs["noise.png"]["entropy"]["value"] > outputs["dark.png"]["entropy"]["value"]
    assert outputs["noise.png"]["edge_density"]["value"] > outputs["dark.png"]["edge_density"]["value"]
    assert outputs["red.png"]["rgb_stats"]["means"] == {
        "red": 1.0,
        "green": 0.0,
        "blue": 0.0,
    }
    assert outputs["red.png"]["saturation"]["mean"] == 1.0


def test_label_balanced_visual_profile_is_descriptive_only(tmp_path: Path):
    rows = [(f"sample-{index}", index % 2) for index in range(8)]
    _write_metadata(tmp_path, rows)
    for sample_id, label in rows:
        pixels = np.full((16, 16, 3), 30 if label == 0 else 220, dtype=np.uint8)
        _save_rgb(tmp_path / f"{sample_id}.png", pixels)

    result = _profile(tmp_path)
    payload = result.model_dump(mode="json")
    assert result.sampling_strategy == "class_balanced"
    assert result.coverage.decoded_samples == 8
    assert result.coverage.selected_per_group == {"label=0": 4, "label=1": 4}
    assert len(result.group_summaries) == 2
    assert result.descriptor_differences
    assert [item.rank for item in result.descriptor_differences] == list(
        range(1, len(result.descriptor_differences) + 1)
    )
    forbidden = {"candidate", "shortcut_verdict", "p_value", "significant"}
    assert forbidden.isdisjoint(payload)


def test_spatial_and_independent_variation_are_recorded_without_verdict(tmp_path: Path):
    rows = [(f"sample-{index}", index % 2) for index in range(12)]
    _write_metadata(tmp_path, rows)
    rng = np.random.default_rng(42)
    for index, (sample_id, label) in enumerate(rows):
        pixels = rng.integers(0, 35, (20, 20, 3), dtype=np.uint8)
        # A public, observable spatial pattern varies by label. Independent
        # noise also varies by sample and remains a descriptor, not a verdict.
        if label == 0:
            pixels[:4, :4, 0] = 255
        else:
            pixels[-4:, -4:, 2] = 255
        pixels[index % 20, :, 1] = (index * 19) % 256
        _save_rgb(tmp_path / f"{sample_id}.png", pixels)

    result = _profile(tmp_path, max_samples=8)
    names = [item.descriptor for item in result.descriptor_differences]
    assert any("top_left_rgb_mean_red" == name for name in names)
    assert any("bottom_right_rgb_mean_blue" == name for name in names)
    serialized = result.model_dump_json().casefold()
    assert "confirmed" not in serialized
    payload = result.model_dump(mode="json")

    def keys(value):
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    assert "p_value" not in keys(payload)


def test_class_independent_variation_is_retained_as_description(tmp_path: Path):
    rows = [(f"sample-{index}", index % 2) for index in range(8)]
    _write_metadata(tmp_path, rows)
    # Both labels receive the same brightness sequence. Within-group variation
    # is real and retained even though it is not label-specific.
    values = (20, 20, 80, 80, 160, 160, 240, 240)
    for (sample_id, _label), value in zip(rows, values, strict=True):
        Image.new("RGB", (8 + value // 20, 8), (value, value, value)).save(
            tmp_path / f"{sample_id}.png"
        )
    result = _profile(tmp_path)
    assert all(
        summary.descriptor_stds["brightness"] > 0
        for summary in result.group_summaries
    )
    brightness = next(
        item
        for item in result.descriptor_differences
        if item.descriptor == "brightness"
    )
    assert brightness.absolute_difference == 0
    assert brightness.standardized_descriptive_difference == 0


def test_metadata_free_scan_and_corrupt_coverage_are_honest(tmp_path: Path):
    for index in range(8):
        Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(
            tmp_path / f"valid-{index}.png"
        )
    for index in range(2):
        (tmp_path / f"corrupt-{index}.png").write_bytes(
            b"\x89PNG\r\n\x1a\nnot-an-image"
        )
    result = _profile(tmp_path, metadata_file=None, max_samples=10)
    assert result.sampling_strategy == "random"
    assert result.coverage.selected_samples == 10
    assert result.coverage.decoded_samples == 8
    assert result.coverage.failed_samples == 2
    assert result.coverage.selected_per_group == {"ungrouped": 10}
    assert result.group_summaries[0].group == "ungrouped"
    assert not result.descriptor_differences
    assert any("label_grouping_unavailable" in item for item in result.limitations)
    assert all(not item.group_values for item in result.sample_observations)


def test_semantic_mock_contract_and_unavailable_default(tmp_path: Path):
    _write_metadata(tmp_path, [("sample-0", 0)])
    Image.new("RGB", (8, 8), "green").save(tmp_path / "sample-0.png")
    default = _profile(tmp_path, max_samples=1)
    assert default.semantic.requested is False
    assert default.semantic.available is False

    observation = SemanticObservation(
        sample_id="sample-0",
        sample_reference="sample-0.png",
        observation_type="background_context",
        region="full image",
        description="A deterministic mock records a green background.",
        confidence="low",
        limitations=["Mock output is contract-only and not model evidence."],
    )
    mock = MockSemanticVisualInspector({"sample-0": [observation]})
    semantic = _profile(
        tmp_path,
        run_id="semantic-visual-run",
        max_samples=1,
        options=VisualDiscoveryOptions(
            max_samples=1,
            sample_id_column="sample_id",
            sample_path_column="sample_path",
            semantic_enabled=True,
        ),
        semantic_inspector=mock,
    )
    assert semantic.semantic.available is True
    assert semantic.semantic.observations == [observation]
    assert len(mock.calls) == 1
    assert mock.calls[0].samples[0].sample_reference == "sample-0.png"

    request = _request(
        tmp_path, run_id="semantic-visual-run", max_samples=1
    )
    dataset_profile = build_dataset_profile(request, seed=17)
    context = build_identify_context_from_profile(
        request,
        seed=17,
        profile=dataset_profile,
        visual_discovery=semantic,
    )
    report = CandidateReport.model_validate(
        {
            "protocol_version": "1.0",
            "run_id": request.run_id,
            "message_id": f"{request.run_id}:identify:report",
            "source": "identify",
            "target": "executive",
            "message_type": "candidate_report",
            "in_reply_to": request.message_id,
            "artifact_references": [],
            "dataset_summary": {
                "dataset": request.dataset.name,
                "samples_inspected": 1,
                "metadata_used": True,
            },
            "candidates": [
                {
                    "candidate_id": "C001",
                    "cue": {"family": "context", "name": "background_context"},
                    "description": "A structured semantic observation warrants review.",
                    "observation": "The injected semantic contract returned a bounded background observation.",
                    "hypothesis": "Background context may provide an unintended cue.",
                    "evidence_sources": ["sample_inspection"],
                    "priority": "low",
                    "status": "unverified",
                }
            ],
            "limitations": list(context.limitations),
        }
    )
    validate_candidate_report(report, request, context)


def test_resume_reuses_artifact_without_decoding(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_metadata(dataset, [("sample-0", 0), ("sample-1", 1)])
    for index in range(2):
        Image.new("RGB", (8, 8), index * 255).save(
            dataset / f"sample-{index}.png"
        )
    calls = 0

    def counting_inspector(*args, **kwargs):
        nonlocal calls
        calls += 1
        return inspect_image_file(*args, **kwargs)

    profiler = VisualDiscoveryProfiler(image_inspector=counting_inspector)
    first = _profile(
        dataset,
        max_samples=2,
        runs_root=tmp_path / "runs",
        profiler=profiler,
    )
    first_calls = calls
    second = _profile(
        dataset,
        max_samples=2,
        runs_root=tmp_path / "runs",
        profiler=profiler,
    )
    assert first == second
    assert first_calls == 2
    assert calls == first_calls
    artifact = tmp_path / "runs" / first.run_id / "identify" / "visual_discovery.json"
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert str(dataset) not in json.dumps(persisted)
    assert persisted["artifact_references"] == ["identify/visual_discovery.json"]


def test_fingerprint_changes_with_content_or_options(tmp_path: Path):
    _write_metadata(tmp_path, [("sample-0", 0)])
    Image.new("RGB", (8, 8), "black").save(tmp_path / "sample-0.png")
    first = _profile(tmp_path, max_samples=1)
    Image.new("RGB", (8, 8), "white").save(tmp_path / "sample-0.png")
    changed_content = _profile(tmp_path, max_samples=1)
    changed_options = _profile(
        tmp_path,
        max_samples=1,
        options=VisualDiscoveryOptions(
            max_samples=1,
            sample_path_column="sample_path",
            sample_id_column="sample_id",
            requested_descriptors=["dimensions"],
        ),
    )
    assert first.dataset_fingerprint != changed_content.dataset_fingerprint
    assert changed_content.request_fingerprint != changed_options.request_fingerprint


def test_total_byte_budget_and_configuration_guards(tmp_path: Path):
    _write_metadata(tmp_path, [("sample-0", 0), ("sample-1", 1)])
    for index in range(2):
        Image.new("RGB", (32, 32), index * 255).save(
            tmp_path / f"sample-{index}.png"
        )
    file_size = (tmp_path / "sample-0.png").stat().st_size
    result = _profile(
        tmp_path,
        max_samples=2,
        options=VisualDiscoveryOptions(
            max_samples=2,
            sample_id_column="sample_id",
            sample_path_column="sample_path",
            max_total_inspection_bytes=file_size,
        ),
    )
    assert result.coverage.decoded_samples == 1
    assert result.coverage.failed_samples == 1
    assert result.coverage.truncated is True
    assert result.coverage.bytes_inspected <= file_size

    request = _request(tmp_path, max_samples=1)
    dataset_profile = build_dataset_profile(request, seed=17)
    with pytest.raises(VisualDiscoveryError, match="exceeds IdentifyBudget"):
        VisualDiscoveryProfiler().profile(
            request,
            dataset_profile,
            seed=17,
            options=VisualDiscoveryOptions(max_samples=2),
        )


def test_protocol_is_strict_and_backend_has_no_candidate_construction():
    with pytest.raises(ValidationError):
        VisualDiscoveryOptions.model_validate({"magic_threshold": 0.5})
    source = Path("shortcutminer/experiment/visual_discovery.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not {"Candidate", "CandidateReport", "EvaluationVerdict"} & imported
    assert not {"Candidate", "CandidateReport", "EvaluationVerdict"} & called
    assert "magic_threshold" not in source
    assert DESCRIPTOR_DEFINITION_VERSION == "image_descriptor_v1"

    production_text = "\n".join(
        path.read_text(encoding="utf-8").casefold()
        for path in (
            Path("shortcutminer/experiment/visual_discovery.py"),
            Path("shortcutminer/inspection/image.py"),
            Path("shortcutminer/inspection/semantic.py"),
            Path("shortcutminer/protocol/visual_discovery.py"),
            Path("shortcutminer/storage/visual_discovery.py"),
        )
    )
    forbidden_benchmarks = {
        "waterbirds",
        "colored-mnist",
        "colored_mnist",
        "celeba",
        "camelyon",
        "hidden_truth",
        "private_truth",
    }
    assert not any(token in production_text for token in forbidden_benchmarks)


def test_shared_flattening_matches_visual_profile(tmp_path: Path):
    _write_metadata(tmp_path, [("sample-0", 0)])
    Image.new("RGB", (10, 12), (200, 20, 10)).save(tmp_path / "sample-0.png")
    result = _profile(tmp_path, max_samples=1)
    observation = result.sample_observations[0]
    assert observation.descriptors is not None
    assert observation.descriptors.values == flatten_image_descriptors(
        observation.descriptors.structured,
        file_size_bytes=observation.file_size_bytes,
    )


def test_visual_evidence_sources_require_real_decoded_evidence(tmp_path: Path):
    _write_metadata(tmp_path, [("sample-0", 0), ("sample-1", 1)])
    for index in range(2):
        Image.new("RGB", (8, 8), index * 255).save(
            tmp_path / f"sample-{index}.png"
        )
    request = _request(tmp_path, max_samples=2)
    dataset_profile = build_dataset_profile(request, seed=17)
    visual = VisualDiscoveryProfiler().profile(
        request,
        dataset_profile,
        seed=17,
        options=VisualDiscoveryOptions(
            max_samples=2,
            sample_id_column="sample_id",
            sample_path_column="sample_path",
        ),
    )
    assert visual is not None
    context = build_identify_context_from_profile(
        request,
        seed=17,
        profile=dataset_profile,
        visual_discovery=visual,
    )
    data = {
        "protocol_version": "1.0",
        "run_id": request.run_id,
        "message_id": f"{request.run_id}:identify:report",
        "source": "identify",
        "target": "executive",
        "message_type": "candidate_report",
        "in_reply_to": request.message_id,
        "artifact_references": [],
        "dataset_summary": {
            "dataset": request.dataset.name,
            "samples_inspected": 2,
            "metadata_used": True,
        },
        "candidates": [
            {
                "candidate_id": "C001",
                "cue": {"family": "appearance", "name": "brightness"},
                "description": "Brightness differs descriptively between selected groups.",
                "observation": "Decoded image descriptors contain different group means.",
                "hypothesis": "Brightness may be an unintended predictive cue.",
                "evidence_sources": ["sample_inspection", "image_statistics"],
                "priority": "medium",
                "status": "unverified",
            }
        ],
        "limitations": list(context.limitations),
    }
    report = CandidateReport.model_validate(data)
    validate_candidate_report(report, request, context)

    no_visual = build_identify_context_from_profile(
        request, seed=17, profile=dataset_profile
    )
    invalid = report.model_copy(deep=True)
    invalid.dataset_summary.samples_inspected = 0
    with pytest.raises(IdentifyExecutionError, match="sample inspection"):
        validate_candidate_report(invalid, request, no_visual)

    visual_without_descriptors = visual.model_copy(deep=True)
    visual_without_descriptors.sample_observations = []
    visual_without_descriptors.coverage.selected_samples = 0
    visual_without_descriptors.coverage.available_samples = 0
    visual_without_descriptors.coverage.decoded_samples = 0
    visual_without_descriptors.coverage.failed_samples = 0
    visual_without_descriptors.coverage.selected_per_group = {}
    context_without_descriptors = build_identify_context_from_profile(
        request,
        seed=17,
        profile=dataset_profile,
        visual_discovery=visual_without_descriptors,
    )
    invalid_image_statistics = report.model_copy(deep=True)
    invalid_image_statistics.dataset_summary.samples_inspected = 0
    invalid_image_statistics.candidates[0].evidence_sources = ["image_statistics"]
    with pytest.raises(IdentifyExecutionError, match="image statistics"):
        validate_candidate_report(
            invalid_image_statistics, request, context_without_descriptors
        )
