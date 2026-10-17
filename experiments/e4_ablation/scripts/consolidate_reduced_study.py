"""Reference retained results without rewriting them or fabricating truth metrics."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.scripts.collect_results import collect_arm
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.autonomous_audit import sha


def consolidate(study):
    study=Path(study).resolve();p=json.loads((study/'study.json').read_text(encoding='utf-8'))
    rows=[]
    for a in p['arms']+p['retained_arms']:
        if 'status_sha256' in a and sha(Path(a['directory'])/'arm-status.json')!=a['status_sha256']:
            raise ValueError('A retained result changed')
        row=collect_arm(a)
        row['origin']='retained study-v2' if 'status_sha256' in a else 'new '+p['protocol_revision']
        row['origin_directory']=a['directory']
        row['summary_sha256']=sha(Path(row['final_directory'])/'summarize/result.json') if (Path(row['final_directory'])/'summarize/result.json').exists() else None
        rows.append(row)
    rows.sort(key=lambda r:(r['case'],p['selected_variants'].index(r['variant'])))
    full=[]
    for case,directory in p['historical_full'].items():
        directory=Path(directory);summary=directory/'summarize/result.json'
        s=json.loads(summary.read_text(encoding='utf-8'))['payload']
        full.append({'case':case,'directory':str(directory),'summary_sha256':sha(summary),
            'assessments':{f['candidate_id']:f['assessment'] for f in s.get('findings',[])},
            'comparison_status':'historical reference; input/config/source differences require disclosure; no matched-control equivalence asserted'})
    write(study/'comparison.json',{'arms':rows,'historical_full':full,'scope':p['scope_decision'],
        'compatibility':p['compatibility'],'truth_policy':'No hidden truth is supplied to auditing. Scientific confirmation is not inferred from runtime completion. Preserve independent adjudication and unknown candidate truth separately.'})
    fields=['case','variant','status','origin','candidate_count','validation_success_rate','known_total_tokens','token_usage_unknown_calls','wall_seconds','final_directory','summary_sha256']
    with (study/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines=['# E4 Reduced Comparison','','Scope: E2-A and E2-B; five variants per dataset. Eight results are retained by reference; only two E2-A baselines are rerun. Full uses historical reports. Counterevidence ablation, E2-C and E3 are outside this revised scope. The scope reduction was decided after the original execution outcomes were inspected.','','## Selected results','','| Dataset | Variant | Execution | Provenance | Candidates | Model tested | Paired OOD tested | Agent assessments |','| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for r in rows:
        findings=', '.join(k+': '+v for k,v in r['assessments'].items()) or 'not yet available'
        lines.append(f"| {r['case']} | {r['variant']} | {r['status']} | {r['origin']} | {r['candidate_count']} | {len(r['model_tested_candidates'])} | {len(r['paired_ood_tested_candidates'])} | {findings} |")
    lines+=['','## Historical Full references','']
    for r in full:
        lines.append('- '+r['case']+': ['+Path(r['directory']).name+']('+r['directory'].replace('\\','/')+'/summarize/result.json). '+str(r['assessments']))
    lines+=['','## Compatibility and interpretation','','- The reused eight runs are unchanged, not relabeled as having run under the recovery source. Their status hashes are checked during consolidation. All paths and final summary hashes are in comparison.json.',
        '- The two baseline reruns use the original frozen public inputs, exact profile and common budgets. The E4 IO protocol revision is recorded separately; no validator or scientific gate is bypassed. Original failed attempts remain preserved and are not erased from reliability or development history.',
        '- Historical Full is not automatically a matched concurrent control: earlier prompts, construction policies, source and memory may differ. Preserve its original assessment and do not claim a causal ablation effect from unexamined cross-version differences.',
        '- Completed means execution finished, not that a shortcut was confirmed. Partial support and unknown natural-candidate truth are not promoted to verified discoveries. Known token totals exclude unavailable usage; classifier seeds are not independent audit repetitions.',
        '- Prior E2 post-run review: ['+str(Path(p['previous_study'])/'independent-evaluation.md').replace('\\','/')+']('+str(Path(p['previous_study'])/'independent-evaluation.md').replace('\\','/')+'). It applies to the old reviewed cells; new baseline candidates require their own post-run scoring.',
        '','## Result files','']
    for r in rows:lines.append('- '+r['case']+'/'+r['variant']+': ['+r['status']+']('+r['final_directory'].replace('\\','/')+'/summarize/result.json)')
    (study/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'selected':len(rows),'completed':sum(r['status']=='COMPLETED' for r in rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    consolidate(p.parse_args().study)
