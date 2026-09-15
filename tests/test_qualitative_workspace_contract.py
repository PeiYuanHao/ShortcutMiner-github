from pathlib import Path


WORKSPACE = (
    Path(__file__).resolve().parents[1]
    / "agents"
    / "qualitative"
    / "workspace"
)


def test_agent_contract_matches_qualitative_protocol():
    instructions = (WORKSPACE / "AGENTS.md").read_text(encoding="utf-8")

    for required_field in (
        "source: orchestrator",
        "evidence_context:",
        "in_reply_to:",
        "seed:",
        "inspection_quality:",
        "failures:",
        "artifact_references:",
    ):
        assert required_field in instructions

    assert "artifact_refs:" not in instructions
    assert "serialization may vary" not in instructions
    for literal_contract in (
        "supported |",
        "partially_supported |",
        "inconclusive |",
        "contradicted |",
        "insufficient_evidence",
        "inspection_quality: sufficient | limited | unusable",
        "confidence: high | medium | low",
        "missing_samples | missing_metadata |",
        "inspection_failure | tool_failure",
        "source: qualitative",
        "target: summarize",
        "message_type: qualitative_analysis_result",
        "QO-<candidate_id>-001",
        "using `[]` when no",
    ):
        assert literal_contract in instructions


def test_tool_contract_preserves_control_layer_evidence_boundary():
    instructions = (WORKSPACE / "TOOLS.md").read_text(encoding="utf-8")

    assert "Do not resample" in instructions
    assert "allowed_output_artifact_prefix" in instructions
    assert "qualitative/<task_id>" in instructions
    assert "inspect_sample" in instructions
    assert "descriptor difference != causal shortcut" in instructions
