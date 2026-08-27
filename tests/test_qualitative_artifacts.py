import json
from pathlib import Path

import pytest

from shortcutminer.protocol.qualitative import QualitativeResult
from shortcutminer.storage.qualitative_artifacts import (
    QualitativeArtifactError,
    get_qualitative_run_dir,
    save_qualitative_input,
    save_qualitative_raw_output,
    save_qualitative_result,
)


def _insufficient_result(run_id: str = "qualitative_test_run") -> QualitativeResult:
    return QualitativeResult(
        protocol_version="1.0",
        run_id=run_id,
        message_id="qualitative:result:C001",
        source="qualitative",
        target="summarize",
        message_type="qualitative_analysis_result",
        in_reply_to="qualitative:request:C001",
        payload={
            "candidate_id": "C001",
            "candidate_name": "dark_border",
            "hypothesis": "A dark border may be associated with the label.",
            "evidence_status": "insufficient_evidence",
            "evidence_summary": "No samples were available.",
            "sampling": {
                "strategy": "group_balanced",
                "seed": 42,
                "groups": {},
                "sample_count": 0,
                "sample_ids": [],
            },
            "observations": [],
            "interpretations": [],
            "counterevidence": [],
            "metadata_evidence": [],
            "failures": [
                {
                    "category": "missing_samples",
                    "description": "No samples were available.",
                    "sample_refs": [],
                    "artifact_references": [],
                }
            ],
            "limitations": ["No samples were available."],
            "artifact_references": [],
            "recommended_followup": ["Provide sample paths."],
        },
    )


def test_qualitative_artifact_layout(tmp_path: Path):
    input_path = save_qualitative_input(
        tmp_path,
        run_id="qualitative_test_run",
        payload={"request": {"run_id": "qualitative_test_run"}},
    )
    raw_path = save_qualitative_raw_output(
        tmp_path,
        run_id="qualitative_test_run",
        raw_output={"message_type": "qualitative_analysis_result"},
    )
    result_path = save_qualitative_result(tmp_path, _insufficient_result())

    expected = tmp_path / "qualitative_test_run" / "qualitative"
    assert input_path == expected / "input.json"
    assert raw_path == expected / "raw_output.json"
    assert result_path == expected / "result.json"
    assert json.loads(result_path.read_text("utf-8"))["target"] == "summarize"


@pytest.mark.parametrize("run_id", ["..", "../escape", "nested/run", r"nested\run", "CON"])
def test_unsafe_run_ids_are_rejected(tmp_path: Path, run_id: str):
    with pytest.raises(QualitativeArtifactError, match="run_id"):
        get_qualitative_run_dir(tmp_path, run_id)
