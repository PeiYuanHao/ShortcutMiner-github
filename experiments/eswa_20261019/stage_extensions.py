"""Offline main-run accounting and pre-outcome extension scope selection; no model calls."""
import csv
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.waterbirds_paper import sha


def main():
    study=ROOT/'results_eswa_single';directory=study/'extensions'
    directory.mkdir(exist_ok=True)
    target=directory/'SCOPE_PLAN.json'
    if target.exists():raise ValueError('Extension scope already planned; preserve and inspect it')
    freeze_dir=study/'protocol/formal-main-v1'
    frozen=json.loads((freeze_dir/'FREEZE.json').read_text())
    for name,key in [('source.zip','source_archive_sha256'),('profile.json','profile_sha256'),
        ('schedule.json','schedule_sha256'),('rules.json','rules_sha256')]:
        if sha(freeze_dir/name)!=frozen[key]:raise ValueError('Main frozen artifact changed')
    with (study/'FORMAL_TASK_STATUS.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if len(rows)!=21 or any(r['status'] not in {'COMPLETED','FAILED_RETAINED'} for r in rows):raise ValueError('Main comparison not exhausted')
    # Scope selection uses only predeclared private task metadata and costs, never effect sizes or winners.
    public=json.loads((study/'FORMAL_PUBLIC_INPUTS.json').read_text())
    selected=[]
    for case in public['cases']:
        private_path=study/'private/formal'/(case['case_id']+'.json')
        spec=json.loads(private_path.read_text())['spec']
        if spec['label_cue_match_rho']==0.95 and spec['mechanism'] in {'watermark','border'}:
            selected.append({'case_id':case['case_id'],'private_record_sha256':sha(private_path),
                'discovery_source':str(study/'audits/formal'/case['case_id']/'full/attempt-01'/(case['case_id']+'-full-discovery')),
                'selection':'Predeclared strong border/watermark tasks, independent of measured outcomes.'})
    if len(selected)!=2:raise ValueError('Expected two predeclared strong-task blocks')
    known=reserved=0;costs=[];unknown_requests=0
    for path in (study/'audits').glob('*/*/*/attempt-*/EXECUTION_RECEIPT.json'):
        receipt=json.loads(path.read_text());new=receipt.get('new_tokens',receipt['known_tokens'])
        reserve=receipt.get('reserved_unknown_tokens',0);known+=new;reserved+=reserve
        unknown=[]
        for record in path.parent.glob('*/model_calls/*.json'):
            if record.name.startswith(('telemetry','runtime_snapshot')):continue
            call=json.loads(record.read_text())
            if call.get('invocation_id') and not call.get('usage',{}).get('available',False) and call.get('status')!='budget_exceeded':
                unknown.append({'record':str(record),'record_sha256':sha(record),
                    'runtime_kind':call.get('runtime_kind'),'category':call.get('failure_category')})
        unknown_requests+=len(unknown)
        costs.append({'receipt':str(path),'receipt_sha256':sha(path),'phase':receipt['phase'],
            'case_id':receipt['case_id'],'variant':receipt['variant'],'new_known_tokens':new,
            'reserved_unknown_estimate':reserve,'usage_incomplete':receipt['usage_incomplete'],
            'wall_seconds':receipt['wall_seconds'],'unknown_requests':unknown})
    cap=600000;ablation_cap=6*cap;pet_cap=cap;headroom=200000
    total=known+reserved+ablation_cap+pet_cap+headroom
    if total>16000000:raise ValueError('Whole-block extension scope does not fit study planning envelope')
    start=json.loads((study/'STUDY_START.json').read_text())
    deadline=datetime.fromisoformat(start['experiment_deadline_utc'])
    now=datetime.now(timezone.utc)
    if now+timedelta(seconds=7*4500)>deadline-timedelta(hours=12):raise ValueError('Insufficient time for extension blocks and export')
    plan={'recorded_utc':now.isoformat(),'status':'SCOPE_SELECTED_HARNESS_NOT_YET_FROZEN',
        'main_source_tree_sha256':frozen['source_tree_sha256'],'main_reruns_authorized':False,
        'ablation_task_blocks':selected,'ablation_arms':['full_same_candidates','without_natural_language_semantic_feedback','fixed_routing'],
        'ablation_audits':6,'pet_audits':1,'per_audit_known_plus_reserved_stop':cap,
        'per_audit_wall_seconds':4500,'known_spent':known,'reserved_unknown_estimates':reserved,
        'planned_new_call_caps':ablation_cap+pet_cap,'global_headroom':headroom,
        'planning_total':total,'unreported_usage_request_count':unknown_requests,
        'billing_limitations':'Known tokens are observed usage; conservative reservations are estimates. Missing historical/multimodal billing remains unknown, not zero. The16M envelope is planning, not a verified invoice cap.',
        'scope_reduction':'Follow predeclared five-day downgrade: omit patch ablation and patch extra seeds; retain complete three-arm border/watermark blocks plus Pet. Scope selected from metadata/costs before reviewing effect sizes.',
        'fixed_candidate_entry':'Freeze the same training-only Full discovery candidate list and measurements for every arm. Full must rerun the controlled entry. Never substitute end-to-end main Full. Construction may vary as part of the feedback mechanism; discoveries must not be regenerated.',
        'feedback_arm_specification':'Retain both branches, images, numerical tools, categorical safety decisions and original validators. Withhold only free-text Qualitative/Executive feedback to construction revision/downstream agents, recording the transmitted mask and retaining original reviews. Freeze exact fields and tests before calls.',
        'fixed_routing_arm_specification':'Same candidates, tools and thresholds; capability-checked frozen routing order. Do not replace discovery or disable visual tools.',
        'pet_scope':'One Full audit and feasible local background/text intervention diagnostics; unsupported constructions defer. Old low-pass result is texture sensitivity only. No new independent confirmation claim without a real unused pool.',
        'human_review':'PENDING; no fabricated adjudication.',
        'extra_seeds':'Only post-audit frozen-intervention checks for predeclared strong border/watermark if bounded local compute fits. Same checkpoint/recipe identities; not Agent repetitions.',
        'next_action':'Implement/test the fixed-candidate extension and Pet entrypoints, freeze a separate source/protocol uniformly for extension arms, then start the bounded resumable extension queue. Main source archive and results remain immutable.'}
    write(target,plan)
    write(directory/'COST_LEDGER.json',{'known_tokens':known,'reserved_unknown_estimates':reserved,
        'unreported_usage_requests':unknown_requests,'receipts':costs})
    for row in rows:
        path=study/'audits/formal'/row['case_id']/row['variant']/'attempt-01/EXECUTION_RECEIPT.json'
        receipt=json.loads(path.read_text())
        if receipt['source_tree_sha256']!=frozen['source_tree_sha256'] or receipt['profile_sha256']!=frozen['profile_sha256']:
            raise ValueError('Main row source/profile identity mismatch')
    write(directory/'MAIN_EXECUTION_REVIEW.json',{'reviewed_utc':now.isoformat(),
        'scheduled':21,'completed':sum(r['status']=='COMPLETED' for r in rows),
        'failed_retained':sum(r['status']=='FAILED_RETAINED' for r in rows),
        'source_profile_match':True,'effects_reviewed':False,'rows':rows,
        'interpretation':'Execution accounting only; failures remain in denominator, not negative scientific results. Full completion is not efficacy.'})
    print(json.dumps({'main_completed':13,'main_failed_retained':8,'known_tokens':known,
        'reserved_unknown_estimates':reserved,'extension_audits':7,'planning_total':total,
        'scope_plan':str(target),'harness_frozen':False}))


if __name__=='__main__':main()
