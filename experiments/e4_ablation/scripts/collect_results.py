"""Read-only E4 evidence/cost accounting. Never manufacture independent truth."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write


def load(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def collect_arm(arm):
    root=Path(arm['directory']);status=load(root/'arm-status.json',{})
    final=Path(status.get('final_directory',root/arm['run_id']))
    summary=load(final/'summarize/result.json',{}).get('payload',{})
    report=load(final/'identify/output.json',{})
    ledger=load(final/'evidence/ledger.json',{}).get('candidates',[])
    candidates={c['candidate_id']:c for c in report.get('candidates',[])}
    assessments={f['candidate_id']:f['assessment'] for f in summary.get('findings',[])}
    coverage={};validated=set();model_tested=set();paired_tested=set();active_counter=set()
    for c in ledger:
        records=c.get('records',[]);cid=c['candidate_id']
        usable=[r for r in records if r['execution_state']=='usable']
        if usable:validated.add(cid)
        stat=any(r['evidence_kind']=='quantitative_association' for r in usable)
        experiment=any(r['evidence_kind'] in {'quantitative_model_reliance','quantitative_ood','quantitative_counterfactual','quantitative_intervention'} for r in usable)
        if experiment:model_tested.add(cid)
        if any(r['evidence_kind'] in {'quantitative_ood','quantitative_counterfactual'} for r in usable):paired_tested.add(cid)
        counter=any(r['evidence_kind']=='qualitative_counterevidence' or r['direction']=='contradicts' for r in usable)
        limitation=bool(candidates.get(cid,{}).get('limitations')) or any(r.get('limitations') for r in records)
        coverage[cid]={'statistical':stat,'experiment':experiment,'observed_counterevidence':counter,'limitations':limitation}
    for p in root.glob('*/executive/routing.json'):
        for task in load(p,{}).get('tasks',[]):
            if 'counterexample_inspection' in task.get('requested_checks',[]):active_counter.add(task['candidate_id'])
    calls=[];ids=set()
    for p in root.glob('*/model_calls/*.json'):
        item=load(p)
        if not item.get('invocation_id'):continue
        if item['invocation_id'] not in ids:ids.add(item['invocation_id']);calls.append(item)
    usages=[c['usage'] for c in calls]
    known=sum(u.get('total_tokens') or 0 for u in usages if u.get('available'))
    unknown=sum(not u.get('available') for u in usages)
    n=len(candidates)
    return {'case':arm['case'],'variant':arm['variant'],'status':status.get('status','PENDING'),
        'candidate_count':n,'assessments':assessments,'candidates':list(candidates.values()),
        'successfully_validated_candidates':sorted(validated),'model_tested_candidates':sorted(model_tested),
        'paired_ood_tested_candidates':sorted(paired_tested),
        'validation_success_rate':len(validated)/n if n else None,
        'evidence_completeness':coverage,'active_counterexample_candidates':sorted(active_counter),
        'text_provider_calls':sum(c['runtime_kind']=='text_agent' and c['status']!='budget_exceeded' for c in calls),
        'visual_provider_calls':sum(c['runtime_kind']=='semantic_vlm' and c['status']!='budget_exceeded' for c in calls),
        'known_total_tokens':known,'token_usage_unknown_calls':unknown,
        'total_tokens':None if unknown else known,'tool_calls':status.get('actual_tool_calls'),
        'wall_seconds':status.get('wall_seconds'),'final_directory':str(final),
        'discovery_recall':None,'verified_discovery_precision':None,'false_positive_rate':None,
        'truth_metric_status':'Pending independent candidate-to-truth matching; E3 has no human adjudication. Unmatched natural cues remain unknown.',
        'independently_adjudicated_candidate_count':0,'not_independently_adjudicated_candidate_count':n}


def collect(study):
    study=Path(study);protocol=load(study/'study.json');rows=[collect_arm(a) for a in protocol['arms']]
    write(study/'results.json',{'schema_version':'e4-results-v1','arms':rows,
        'warning':'Runtime completion and model sensitivity do not establish independent shortcut truth. Null values are not zeros.'})
    with (study/'results.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['case','variant','status','candidate_count','validation_success_rate','text_provider_calls','visual_provider_calls',
            'total_tokens','known_total_tokens','token_usage_unknown_calls','tool_calls','wall_seconds',
            'discovery_recall','verified_discovery_precision','false_positive_rate']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# E4 Experiment Progress and Results','','One independent discovery audit per arm; task-model seeds are conditional internal replicates.',
        'E3 has no frozen human adjudication. Independent truth metrics remain unavailable until post-run matching is recorded.',
        'E2-C is a label-independent watermark control, not a watermark-free dataset.','','| Input | Variant | Status | Candidates | Validated | Model tested | Known tokens | Minutes |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        elapsed=f"{r['wall_seconds']/60:.1f}" if r['wall_seconds'] is not None else ''
        lines.append(f"| {r['case']} | {r['variant']} | {r['status']} | {r['candidate_count']} | {len(r['successfully_validated_candidates'])} | {len(r['model_tested_candidates'])} | {r['known_total_tokens']} | {elapsed} |")
    (study/'progress.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    rows=collect(p.parse_args().study);print(json.dumps({'statuses':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}}))
