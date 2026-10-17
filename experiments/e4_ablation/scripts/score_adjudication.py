"""External post-run scoring from independently frozen candidate adjudication."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write
from experiments.e4_ablation.scripts.collect_results import collect


def score(row, judgment):
    candidates={c['candidate_id'] for c in row['candidates']}
    entries=judgment['candidates']
    if set(entries)-candidates:raise ValueError('Unknown candidate in independent adjudication')
    if judgment.get('prior_run_automatically_used_as_truth'):raise ValueError('A historical run is not independent truth')
    true={cid for cid,v in entries.items() if v['verdict']=='true' and v.get('independent_validation_reference')}
    false={cid for cid,v in entries.items() if v['verdict']=='false'}
    unknown=candidates-true-false
    planted=set(judgment.get('positive_mechanism_ids',[]));matched=set()
    for cid,v in entries.items():
        if v.get('discovery_match') and v.get('mechanism_id') in planted:matched.add(v['mechanism_id'])
    confirmed={cid for cid,a in row['assessments'].items() if a=='supported'}
    negative=set(judgment.get('negative_mechanism_ids',[]))
    false_confirmed={v['mechanism_id'] for cid,v in entries.items() if cid in confirmed and v.get('mechanism_id') in negative}
    n=len(candidates);adjudicated=len(true|false)
    return {'case':row['case'],'variant':row['variant'],
        'discovery_recall':len(matched)/len(planted) if planted else None,
        'verified_discovery_precision':len(true)/n if n and not unknown else None,
        'precision_among_adjudicated':len(true)/adjudicated if adjudicated else None,
        'precision_lower_bound':len(true)/n if n else None,
        'precision_upper_bound':(len(true)+len(unknown))/n if n else None,
        'adjudication_coverage':adjudicated/n if n else None,
        'unknown_candidates':sorted(unknown),
        'false_positive_rate_on_declared_negative_mechanisms':len(false_confirmed)/len(negative) if negative else None,
        'negative_control_units':len(negative),
        'limitations':['A planted nuisance is not automatically learned by a trained classifier. Independent validation references must substantiate true-candidate verdicts.',
            'Negative-control FPR concerns declared mechanisms only; other natural cues may have unknown truth.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--adjudication',type=Path,required=True);args=p.parse_args()
    raw=args.adjudication.read_bytes();judgments=json.loads(raw)
    if not judgments.get('frozen_at') or not judgments.get('evaluator'):raise ValueError('Record evaluator and freeze time')
    rows=collect(args.study);results=[]
    for row in rows:
        key=row['case']+'/'+row['variant']
        if key not in judgments['arms']:continue
        if row['status']!='COMPLETED':raise ValueError('Adjudication requires a completed arm')
        results.append(score(row,judgments['arms'][key]))
    write(args.study/'independent-scores.json',{'adjudication_sha256':hashlib.sha256(raw).hexdigest(),'scores':results})


if __name__=='__main__':main()
