import json

from shortcutminer.experiment.audit_report import render_audit_markdown, SCREENING_EXPLANATION, build_audit_report
from shortcutminer.protocol.audit_report import AuditReport
from test_audit_report import completed_audit


def test_markdown_renders_same_canonical_json(completed_audit):
    _, _, root, _ = completed_audit
    report = AuditReport.model_validate_json((root / "audit_report.json").read_text(encoding="utf-8"))
    assert (root / "audit_report.md").read_text(encoding="utf-8") == render_audit_markdown(report)
    markdown = render_audit_markdown(report)
    assert f"**{SCREENING_EXPLANATION}**" in markdown
    for title in ("Run Summary", "Dataset", "Shortcut Candidate Discovery", "Candidate Screening", "Validation Plan",
                  "Quantitative Evidence", "Qualitative Evidence", "Evidence Synthesis", "Final Findings",
                  "Runtime and Cost", "Limitations", "Reproducibility", "Artifact Index"):
        assert f"## {title}" in markdown


def test_markdown_missing_values_show_na(completed_audit):
    request, _, root, _ = completed_audit
    (root / "model_calls/telemetry.json").unlink()
    markdown = render_audit_markdown(build_audit_report(request.runs_root, request.run_id))
    assert "**total_tokens**: N/A" in markdown
    assert "**git_commit**: N/A" in markdown


def test_markdown_scientific_ids_and_assessments_preserved(completed_audit):
    _, _, root, _ = completed_audit
    report = json.loads((root / "audit_report.json").read_text())
    markdown = (root / "audit_report.md").read_text(encoding="utf-8")
    for finding in report["findings"]["findings"]:
        assert finding["candidate_id"] in markdown and finding["assessment"] in markdown
        for identity in finding["evidence_record_ids"]:
            assert identity in markdown
