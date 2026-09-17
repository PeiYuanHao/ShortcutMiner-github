"""Ownership and truth-firewall tests for qualitative finalization."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shortcutminer.experiment.qualitative_backend import sample_has_inspection_evidence
from shortcutminer.experiment.qualitative_context import build_qualitative_context
from shortcutminer.experiment.qualitative_executor import (
    QualitativeDraftValidationError,
    finalize_qualitative_draft,
    validate_qualitative_result,
)
from shortcutminer.evaluation.evidence_ledger import _qualitative_records
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.qualitative import QualitativeAgentDraft, QualitativeResult
from shortcutminer.protocol.quantitative import QuantitativeAgentDraft
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.runtime.openai_compatible import canonical_json_hash
from shortcutminer.tooling.agent_loop import _FINAL_SCHEMAS, _validate_provider_output
from test_qualitative_context import _dataset, _request


def _context_with_numeric_groups(tmp_path: Path):
    _dataset(tmp_path)
    request = _request(tmp_path)
    context = build_qualitative_context(request)
    samples = [
        sample.model_copy(update={"group": group})
        for sample, group in zip(context.selected_samples[:3], ("0", "0", "1"))
    ]
    assert all(sample_has_inspection_evidence(sample) for sample in samples)
    context = context.model_copy(
        update={
            "selected_samples": samples,
            "samples_selected": 3,
            "sample_files_available": 3,
            "samples_inspected": 3,
            "failed_inspections": 0,
        }
    )
    return request, context


def _draft_dict() -> dict:
    return {
        "sample_ids": ["S1", "S2", "S3"],
        "evidence_status": "partially_supported",
        "evidence_summary": "Inspected evidence contains a bounded group difference.",
        "observations": [
            {
                "sample_refs": ["S1", "S2"],
                "observation": "The inspected descriptors share a visible property.",
                "inspection_quality": "sufficient",
            },
            {
                "sample_refs": ["S3"],
                "observation": "The comparison descriptor differs from the first group.",
                "inspection_quality": "sufficient",
            },
        ],
        "comparisons": [
            {
                "sample_refs": ["S1", "S3"],
                "comparison": "The inspected groups differ in the measured property.",
                "inspection_quality": "sufficient",
            }
        ],
        "interpretations": [
            {
                "interpretation": "The difference is descriptive, not causal evidence.",
                "supporting_observation_indexes": [0, 1],
                "supporting_comparison_indexes": [0],
                "confidence": "low",
            }
        ],
        "counterevidence": [
            {
                "sample_refs": ["S2"],
                "description": "Within-group variation limits the conclusion.",
            }
        ],
        "metadata_evidence": [
            {
                "field": "label",
                "observation": "Both authorized labels are represented.",
                "sample_refs": ["S1", "S2", "S3"],
            }
        ],
        "failures": [],
        "additional_limitations": ["Only a bounded subset was inspected."],
        "artifact_references": [],
        "recommended_followup": ["Inspect another deterministic sample."],
    }


def _draft() -> QualitativeAgentDraft:
    return QualitativeAgentDraft.model_validate(_draft_dict())


def test_a_sampling_groups_and_count_are_deterministically_derived(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path)

    result = finalize_qualitative_draft(_draft(), request, context)

    assert result.payload.sampling.groups == {"0": 2, "1": 1}
    assert result.payload.sampling.sample_count == 3


def test_b_provider_draft_forbids_sampling_bookkeeping():
    value = _draft_dict()
    value["sampling"] = {"groups": {"0": 2, "1": 1}}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QualitativeAgentDraft.model_validate(value)


def test_c_unknown_sample_is_rejected(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path)
    value = _draft_dict()
    value["sample_ids"] = ["S1", "S2", "UNKNOWN"]

    with pytest.raises(QualitativeDraftValidationError) as raised:
        finalize_qualitative_draft(
            QualitativeAgentDraft.model_validate(value), request, context
        )

    assert raised.value.category == "unknown_sample_reference"


@pytest.mark.parametrize("inspection_status", ["failed", "completed"])
def test_d_failed_or_uninspected_required_sample_is_rejected(
    tmp_path: Path,
    inspection_status: str,
):
    request, context = _context_with_numeric_groups(tmp_path)
    unavailable = context.selected_samples[2].model_copy(
        update={
            "inspection_status": inspection_status,
            "visual_descriptors": {},
            "text_descriptors": {},
        }
    )
    context = context.model_copy(
        update={
            "selected_samples": [*context.selected_samples[:2], unavailable],
            "samples_inspected": 2,
            "failed_inspections": int(inspection_status == "failed"),
        }
    )

    with pytest.raises(QualitativeDraftValidationError) as raised:
        finalize_qualitative_draft(_draft(), request, context)

    assert raised.value.category == "unavailable_sample_reference"


def test_e_strategy_and_seed_are_preserved_from_context(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path)

    sampling = finalize_qualitative_draft(_draft(), request, context).payload.sampling

    assert sampling.strategy == context.sampling_strategy
    assert sampling.seed == context.seed


def test_f_envelope_and_g_limitations_are_deterministic(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path)

    result = finalize_qualitative_draft(_draft(), request, context)

    assert result.protocol_version == request.protocol_version
    assert result.run_id == request.run_id
    assert result.source == "qualitative"
    assert result.target == "summarize"
    assert result.message_type == "qualitative_analysis_result"
    assert result.in_reply_to == request.message_id
    assert result.message_id not in {request.message_id, request.task.message_id}
    assert result.payload.limitations == [
        *context.limitations,
        "Only a bounded subset was inspected.",
    ]


def test_h_candidate_identity_is_not_provider_owned():
    value = _draft_dict()
    value["candidate_id"] = "C999"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QualitativeAgentDraft.model_validate(value)


def test_i_observation_ids_groups_indexes_and_final_schema(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path)

    result = finalize_qualitative_draft(_draft(), request, context)

    assert [item.observation_id for item in result.payload.observations] == [
        "QO-C001-001",
        "QO-C001-002",
    ]
    assert [item.group for item in result.payload.observations] == ["0", "1"]
    assert result.payload.interpretations[0].supporting_observation_ids == [
        "QO-C001-001",
        "QO-C001-002",
    ]
    assert result.payload.interpretations[0].supporting_comparison_ids == [
        "QC-C001-001"
    ]
    comparison = result.payload.comparisons[0]
    assert comparison.comparison_id == "QC-C001-001"
    assert comparison.groups == ["0", "1"]
    assert comparison.sample_refs == ["S1", "S3"]
    assert comparison.comparison == _draft_dict()["comparisons"][0]["comparison"]
    assert comparison.inspection_quality == "sufficient"
    assert QualitativeResult.model_validate(result.model_dump(mode="json")) == result
    validate_qualitative_result(result, request, context)


def test_observation_cannot_cross_authoritative_groups(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path)
    value = _draft_dict()
    value["observations"][0]["sample_refs"] = ["S1", "S3"]

    with pytest.raises(QualitativeDraftValidationError) as raised:
        finalize_qualitative_draft(
            QualitativeAgentDraft.model_validate(value), request, context
        )

    assert raised.value.category == "observation_crosses_groups"


def test_cross_group_comparison_is_valid_and_groups_follow_first_appearance(
    tmp_path: Path,
):
    request, context = _context_with_numeric_groups(tmp_path)
    value = _draft_dict()
    value["comparisons"][0]["sample_refs"] = ["S3", "S1", "S2"]

    result = finalize_qualitative_draft(
        QualitativeAgentDraft.model_validate(value), request, context
    )

    assert result.payload.comparisons[0].groups == ["1", "0"]


def test_single_group_comparison_is_rejected(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path)
    value = _draft_dict()
    value["comparisons"][0]["sample_refs"] = ["S1", "S2"]

    with pytest.raises(QualitativeDraftValidationError) as raised:
        finalize_qualitative_draft(
            QualitativeAgentDraft.model_validate(value), request, context
        )

    assert raised.value.category == "comparison_single_group"


def test_comparison_unknown_sample_is_rejected(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path)
    value = _draft_dict()
    value["comparisons"][0]["sample_refs"] = ["S1", "UNKNOWN"]

    with pytest.raises(QualitativeDraftValidationError) as raised:
        finalize_qualitative_draft(
            QualitativeAgentDraft.model_validate(value), request, context
        )

    assert raised.value.category == "unknown_sample_reference"


def test_comparison_uninspected_sample_is_rejected(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path)
    unavailable = context.selected_samples[2].model_copy(
        update={
            "inspection_status": "completed",
            "visual_descriptors": {},
            "text_descriptors": {},
        }
    )
    context = context.model_copy(
        update={
            "selected_samples": [*context.selected_samples[:2], unavailable],
            "samples_inspected": 2,
        }
    )
    value = _draft_dict()
    value["observations"] = []
    value["interpretations"] = []

    with pytest.raises(QualitativeDraftValidationError) as raised:
        finalize_qualitative_draft(
            QualitativeAgentDraft.model_validate(value), request, context
        )

    assert raised.value.category == "unavailable_sample_reference"


@pytest.mark.parametrize("extra_field", ["comparison_id", "groups"])
def test_provider_cannot_supply_deterministic_comparison_fields(extra_field: str):
    value = _draft_dict()
    value["comparisons"][0][extra_field] = (
        "QC-C001-001" if extra_field == "comparison_id" else ["0", "1"]
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QualitativeAgentDraft.model_validate(value)


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("supporting_observation_indexes", [2], "unknown observation index"),
        ("supporting_comparison_indexes", [1], "unknown comparison index"),
    ],
)
def test_interpretation_out_of_range_evidence_index_is_rejected(
    tmp_path: Path,
    field: str,
    value: list[int],
    error_match: str,
):
    request, context = _context_with_numeric_groups(tmp_path)
    draft_value = _draft_dict()
    draft_value["interpretations"][0][field] = value

    with pytest.raises(QualitativeDraftValidationError, match=error_match):
        finalize_qualitative_draft(
            QualitativeAgentDraft.model_validate(draft_value), request, context
        )


@pytest.mark.parametrize(
    "field",
    ["supporting_observation_indexes", "supporting_comparison_indexes"],
)
def test_interpretation_duplicate_evidence_indexes_are_rejected(field: str):
    value = _draft_dict()
    value["interpretations"][0][field] = [0, 0]

    with pytest.raises(ValidationError, match="must not contain duplicates"):
        QualitativeAgentDraft.model_validate(value)


def test_j_tool_call_contract_and_k_other_agent_schemas_are_unchanged():
    _validate_provider_output(
        "qualitative",
        {
            "tool_calls": [
                {
                    "name": "inspect_sample_metadata",
                    "arguments": {
                        "sample_id": "S1",
                        "sample_reference": "images/s1.png",
                    },
                }
            ]
        },
    )
    assert _FINAL_SCHEMAS == {
        "identify": CandidateReport,
        "executive": RoutingPlan,
        "quantitative": QuantitativeAgentDraft,
        "qualitative": QualitativeAgentDraft,
        "summarize": SummaryReport,
    }


def test_truth_firewall_keeps_canonical_result_hash_identical(tmp_path: Path):
    request, context = _context_with_numeric_groups(tmp_path / "public")
    truth_path = tmp_path / "private" / "hidden_truth.json"
    truth_path.parent.mkdir()

    truth_path.write_text(json.dumps({"shortcut": "first"}), encoding="utf-8")
    with_truth = finalize_qualitative_draft(_draft(), request, context)
    truth_path.unlink()
    without_truth = finalize_qualitative_draft(_draft(), request, context)
    truth_path.write_text(json.dumps({"shortcut": "replaced"}), encoding="utf-8")
    replaced_truth = finalize_qualitative_draft(_draft(), request, context)

    hashes = {
        canonical_json_hash(item.model_dump(mode="json"))
        for item in (with_truth, without_truth, replaced_truth)
    }
    assert len(hashes) == 1
    ledger_hashes = {
        canonical_json_hash([
            record.model_dump(mode="json")
            for record in _qualitative_records(item, request.task, None)
        ])
        for item in (with_truth, without_truth, replaced_truth)
    }
    assert len(ledger_hashes) == 1
