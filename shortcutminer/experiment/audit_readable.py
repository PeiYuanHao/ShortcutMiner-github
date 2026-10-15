"""Bounded reader-facing view of canonical findings; no new scientific verdicts."""
import html
import json


ASSESSMENTS = {
    'supported': 'Supported', 'partially_supported': 'Partially supported',
    'contradicted': 'Contradicted', 'inconclusive': 'Inconclusive',
    'insufficient_evidence': 'Insufficient evidence',
}
CUE_NAMES = {
    'embedded text watermark or logo overlay': 'Text, watermarks or logos',
    'localized text, logo, and watermark markers': 'Local text, logos or watermarks',
    'global brightness and exposure level': 'Overall brightness and exposure',
    'color balance and saturation profile': 'Color balance and saturation',
    'image complexity and edge/texture density': 'Texture and image detail',
    'image dimensions and aspect ratio': 'Image size and aspect ratio',
}


def _text(value, limit=120):
    value = ' '.join(str(value or 'Not recorded').split())
    if len(value) > limit:
        value = value[:limit] + '…'
    return html.escape(value).replace('|', '\\|').replace('[', '\\[').replace(']', '\\]').replace('`', '\\`')


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _points(value):
    return f'{100*value:.2f}' if _numeric(value) else 'Not recorded'


def _paired_drop(metrics):
    if _numeric(metrics.get('accuracy_drop')):
        return metrics['accuracy_drop']
    if _numeric(metrics.get('pair_count')) and metrics['pair_count']>0 and _numeric(metrics.get('accuracy_delta')):
        return -metrics['accuracy_delta']
    return None


def _recipes(value):
    """Read frozen planned operations only; never inspect external files."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'transformation', 'screening_transformation'} and isinstance(item, dict):
                found.add(json.dumps(item, sort_keys=True, ensure_ascii=False))
            else:
                found.update(_recipes(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_recipes(item))
    return found


def render_readable_audit(report):
    """Display decisions and executed evidence without dumping arrays or references."""
    data = report.model_dump(mode='json')
    findings = data['findings'].get('findings', [])
    candidates = {c['candidate_id']:c for c in data['discovery'].get('candidates', [])}
    quant = {q['payload']['candidate_id']:q['payload'] for q in data['quantitative']}
    qual = {q['payload']['candidate_id'] for q in data['qualitative']
            if q['payload'].get('sampling', {}).get('sample_count', 0) > 0
            and (q['payload'].get('observations') or q['payload'].get('comparisons'))}
    measurements = {cid:[m for m in q.get('measurements', []) if m.get('execution_outcome') == 'completed']
                    for cid,q in quant.items()}
    lines = ['# ShortcutMiner Audit Report', '',
             '> Concise report. Findings come from the Summarize Agent and EvidenceLedger. Full records are available in the technical report.', '',
             '## Findings at a glance', '']
    if findings:
        counts = [f"{label}: {sum(f['assessment']==key for f in findings)}" for key,label in ASSESSMENTS.items()
                  if any(f['assessment']==key for f in findings)]
        lines += [f"{len(findings)} candidates assessed. "+'; '.join(counts)+'.', '']
    else:
        lines += ['No candidate findings were recorded.', '']
    lines += ['Inconclusive or insufficient evidence does not mean shortcuts are absent. A data association alone does not establish model reliance.', '',
              '## Audit scope', '', f"- Run ID: {_text(data['run'].get('run_id'))}",
              f"- Status: {'Workflow completed' if data['run'].get('status')=='COMPLETED' else _text(data['run'].get('status'))}"]
    count = data['dataset'].get('sample_count')
    lines += [f"- Discovery input records: {count if count is not None else 'Not fully counted'}. This is not the total across all validation partitions."]
    elapsed = data['run'].get('elapsed_seconds')
    if _numeric(elapsed): lines += [f'- Recorded workflow duration: approximately {elapsed/60:.1f} minutes.']
    lines += ['', '## Candidate findings and completed checks', '',
              '| Candidate | Suspected cue | Assessment | Checks actually completed |', '| --- | --- | --- | --- |']
    paired = []
    for finding in findings:
        cid = finding['candidate_id']
        cue = candidates.get(cid, {}).get('cue', {}).get('name', cid)
        cue = CUE_NAMES.get(cue.casefold(), cue)
        checks = []
        if cid in qual: checks.append('Image inspection')
        types = {m.get('experiment_type') for m in measurements.get(cid, [])}
        if 'statistical_check' in types: checks.append('Data association check')
        if 'task_model_training' in types: checks.append('Model training or current-run weight reuse')
        if 'task_model_evaluation' in types: checks.append('Original-image evaluation')
        for m in measurements.get(cid, []):
            metrics = m.get('metrics', {})
            if m.get('experiment_type') == 'ood_evaluation' and _paired_drop(metrics) is not None:
                final = metrics.get('final_confirmation_executed')
                checks.append('Paired probe (no independent confirmation)' if final is False else 'Paired intervention evaluation')
                paired.append((cid,m))
        done = '; '.join(dict.fromkeys(checks)) or 'No completed checks recorded'
        lines.append(f"| {_text(cid)} | {_text(cue)} | {ASSESSMENTS.get(finding['assessment'], _text(finding['assessment']))} | {done} |")
    if paired:
        lines += ['', '## Key paired-validation results', '',
                  'The same images are evaluated before and after intervention. A positive drop means worse accuracy after the change; a wider interval means greater uncertainty.', '',
                  '| Candidate | Image pairs | Original accuracy | After intervention | Drop (percentage points) | Uncertainty interval (percentage points) |',
                  '| --- | ---: | ---: | ---: | ---: | --- |']
        for cid,m in paired:
            metrics = m['metrics']
            intervals = [x for x in m.get('uncertainty_metadata', [])
                         if x.get('metric') in {'accuracy_drop','accuracy_delta'} and x.get('available')]
            interval = intervals[0] if intervals else {}
            if interval.get('metric')=='accuracy_delta':
                interval={**interval,'lower':-interval['upper'],'upper':-interval['lower']}
            bounds = f"{_points(interval.get('lower'))} to {_points(interval.get('upper'))}" if interval else 'No available interval'
            before = _points(metrics.get('original_accuracy')); after = _points(metrics.get('counterfactual_accuracy'))
            lines.append(f"| {_text(cid)} | {_text(metrics.get('pair_count', metrics.get('sample_count')))} | {before}{'%' if before!='Not recorded' else ''} | {after}{'%' if after!='Not recorded' else ''} | {_points(_paired_drop(metrics))} | {bounds} |")
        matrix_rows=[]
        for cid,m in paired:
            metrics=m['metrics']
            for key in sorted(metrics):
                if key.endswith('_accuracy_drop_mean') and _numeric(metrics[key]):
                    architecture=key.removesuffix('_accuracy_drop_mean')
                    def value(metric):
                        mean=metrics.get(f'{architecture}_{metric}_mean');std=metrics.get(f'{architecture}_{metric}_std')
                        return _points(mean)+(f' ± {_points(std)}' if _numeric(std) else '')
                    matrix_rows.append(f"| {_text(cid)} | {_text(architecture)} | {value('original_accuracy')} | {value('counterfactual_accuracy')} | {value('accuracy_drop')} |")
        if matrix_rows:
            lines+=['','## Results across recorded seeds','',
                'Mean ± standard deviation across seeds, in percentage points. These rows reuse the same paired images; they are not additional independent samples or confidence intervals. The preceding row is the recorded aggregate model result.','',
                '| Candidate | Architecture | Original accuracy (%) | After intervention (%) | Drop (points) |',
                '| --- | --- | ---: | ---: | ---: |',*matrix_rows]
        # Same operation does not establish independent mechanism-specific tests.
        groups = {}
        for cid,_ in paired:
            task_id = quant[cid].get('task_id')
            for recipe in _recipes(data['validation'].get('plans', {}).get(task_id, {})):
                groups.setdefault(recipe, []).append(cid)
        duplicates = sorted({tuple(sorted(set(ids))) for ids in groups.values() if len(set(ids))>1})
        for ids in duplicates:
            lines += ['', f"Note: {', '.join(map(_text, ids))} use the same intervention operation. These do not independently validate different mechanisms; see the technical report for the operation."]
    lines += ['', '## Important limitations', '']
    if any(m['metrics'].get('final_confirmation_executed') is False for _,m in paired):
        lines += ['- Paired probes were executed, but independent final confirmation was not completed. See the technical report for stopping decisions.']
    if any(not any(m.get('experiment_type')=='ood_evaluation' for m in measurements.get(f['candidate_id'], [])) for f in findings):
        lines += ['- Some candidates have only observational or data-association evidence, without completed model intervention validation.']
    if data['screening'].get('screened_out'):
        lines += ['- Screened-out candidates did not enter subsequent experiments; they were not experimentally disproved.']
    lines += ['- Workflow completion means the audit steps have ended, not that every candidate was successfully validated.',
              '- This page shows key numbers only. Consult the full findings for scope, semantic confounding, sampling limits and follow-up recommendations.', '',
              '## Supporting records', '',
              '- [Full technical report](audit_report.technical.md): procedures, limitations and evidence index.',
              '- [Original Summarize Agent findings](summarize/result.json): candidate-specific explanations and recommendations.',
              '- [Complete structured report](audit_report.json): all metrics and provenance.', '']
    return '\n'.join(lines)
