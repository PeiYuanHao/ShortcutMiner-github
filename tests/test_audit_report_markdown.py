import json

from shortcutminer.experiment.audit_report import render_audit_markdown, render_audit_technical_markdown, SCREENING_EXPLANATION, build_audit_report
from shortcutminer.protocol.audit_report import AuditReport
from test_audit_report import completed_audit


def test_markdown_renders_same_canonical_json(completed_audit):
    _, _, root, _ = completed_audit
    report = AuditReport.model_validate_json((root / "audit_report.json").read_text(encoding="utf-8"))
    assert (root / "audit_report.md").read_text(encoding="utf-8") == render_audit_markdown(report)
    markdown = render_audit_markdown(report)
    for title in ('Findings at a glance', 'Audit scope', 'Candidate findings and completed checks', 'Important limitations'):
        assert f'## {title}' in markdown
    assert 'Record 1:' not in markdown
    assert 'audit_report.technical.md' in markdown
    markdown = render_audit_technical_markdown(report)
    assert (root / 'audit_report.technical.md').read_text(encoding='utf-8') == markdown
    assert f"**{SCREENING_EXPLANATION}**" in markdown
    for title in ("Run Summary", "Dataset", "Shortcut Candidate Discovery", "Candidate Screening", "Validation Plan",
                  "Quantitative Evidence", "Qualitative Evidence", "Evidence Synthesis", "Final Findings",
                  "Runtime and Cost", "Limitations", "Reproducibility", "Artifact Index"):
        assert f"## {title}" in markdown


def test_markdown_missing_values_show_na(completed_audit):
    request, _, root, _ = completed_audit
    (root / "model_calls/telemetry.json").unlink()
    markdown = render_audit_technical_markdown(build_audit_report(request.runs_root, request.run_id))
    assert "**total_tokens**: N/A" in markdown
    assert "**git_commit**: N/A" in markdown


def test_markdown_scientific_ids_and_assessments_preserved(completed_audit):
    _, _, root, _ = completed_audit
    report = json.loads((root / "audit_report.json").read_text())
    markdown = (root / "audit_report.md").read_text(encoding="utf-8")
    technical = (root / 'audit_report.technical.md').read_text(encoding='utf-8')
    for finding in report["findings"]["findings"]:
        assert finding['candidate_id'] in markdown
        assert finding["candidate_id"] in technical and finding["assessment"] in technical
        for identity in finding["evidence_record_ids"]:
            assert identity in technical


def test_readable_report_does_not_expand_evidence_or_invent_confirmation(completed_audit):
    _, _, root, _ = completed_audit
    report = AuditReport.model_validate_json((root / 'audit_report.json').read_text(encoding='utf-8'))
    data = report.model_dump(mode='json')
    data['artifacts']['references'] = [f'images/UNIQUE_BLOB_{i}.png' for i in range(20000)]
    data['findings']['artifact_references'] = data['artifacts']['references']
    # Actual records, rather than planned steps, determine displayed validation.
    cid = data['findings']['findings'][0]['candidate_id']
    data['quantitative'] = [{'payload': {'candidate_id':cid,'task_id':'TASK-TEST',
        'measurements':[{'experiment_type':'ood_evaluation','execution_outcome':'completed',
            'metrics':{'pair_count':1500,'accuracy_drop':0.0093333333,
                'original_accuracy':0.874,'counterfactual_accuracy':0.8646666667,
                'final_confirmation_executed':False},
            'uncertainty_metadata':[{'metric':'accuracy_drop','available':True,
                'lower':-0.005746,'upper':0.025743}]}]}}]
    rendered = render_audit_markdown(AuditReport.model_validate(data))
    assert len(rendered) < 12000
    assert 'UNIQUE_BLOB_' not in rendered
    assert '0.93' in rendered and '-0.57 to 2.57' in rendered
    assert 'no independent confirmation' in rendered and 'independent final confirmation was not completed' in rendered
    assert '87.40%' in rendered
    assert 'confirmed shortcut' not in rendered.lower()
    assert not any('\u4e00' <= char <= '\u9fff' for char in rendered)


def test_readable_does_not_count_planned_or_failed_checks_as_completed(completed_audit):
    _, _, root, _ = completed_audit
    data = json.loads((root / 'audit_report.json').read_text(encoding='utf-8'))
    cid = data['findings']['findings'][0]['candidate_id']
    data['qualitative'] = [{'payload': {'candidate_id':cid, 'sampling':{'sample_count':0},
        'observations':[], 'comparisons':[]}}]
    data['quantitative'] = [{'payload': {'candidate_id':cid, 'measurements':[
        {'experiment_type':'task_model_training','execution_outcome':'blocked','metrics':{}}]}}]
    data['validation']['plans'] = {'task':{'candidate_id':cid, 'requested_checks':['task_model_training','ood_evaluation']}}
    rendered = render_audit_markdown(AuditReport.model_validate(data))
    assert 'No completed checks recorded' in rendered
    assert 'Image inspection' not in rendered
    assert 'Model training or current-run weight reuse' not in rendered
    assert '## Key paired-validation results' not in rendered


def test_final_paired_delta_and_seed_means_are_displayed_without_changing_evidence(completed_audit):
    _,_,root,_=completed_audit
    data=json.loads((root/'audit_report.json').read_text(encoding='utf-8'))
    cid=data['findings']['findings'][0]['candidate_id']
    data['quantitative']=[{'payload':{'candidate_id':cid,'measurements':[{
        'experiment_type':'ood_evaluation','execution_outcome':'completed',
        'metrics':{'pair_count':2169,'original_accuracy':.9267,'counterfactual_accuracy':.8598,'accuracy_delta':-.0669,
            'resnet18_accuracy_drop_mean':.0721,'resnet18_accuracy_drop_std':.0023,
            'resnet18_original_accuracy_mean':.8844,'resnet18_counterfactual_accuracy_mean':.8124},
        'uncertainty_metadata':[{'metric':'accuracy_delta','available':True,'lower':-.0789,'upper':-.0553}]}]}}]
    report=AuditReport.model_validate(data);before=report.model_dump_json()
    rendered=render_audit_markdown(report)
    assert 'Paired intervention evaluation' in rendered
    assert '6.69' in rendered and '5.53 to 7.89' in rendered
    assert '92.67%' in rendered and '85.98%' in rendered
    assert '7.21 ± 0.23' in rendered and 'Results across recorded seeds' in rendered
    assert report.model_dump_json()==before


def test_independent_shift_delta_is_not_presented_as_paired_validation(completed_audit):
    _,_,root,_=completed_audit
    data=json.loads((root/'audit_report.json').read_text(encoding='utf-8'))
    cid=data['findings']['findings'][0]['candidate_id']
    data['quantitative']=[{'payload':{'candidate_id':cid,'measurements':[{
        'experiment_type':'ood_evaluation','execution_outcome':'completed','metrics':{'accuracy_delta':-.1}}]}}]
    rendered=render_audit_markdown(AuditReport.model_validate(data))
    assert '## Key paired-validation results' not in rendered
