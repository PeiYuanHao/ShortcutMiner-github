"""Frozen, serial, bounded main comparison. No outcome-driven retries or extensions."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone,timedelta
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.waterbirds_paper import sha,source_identity,archive_source
from experiments.eswa_20261019.monitor import alive

VARIANTS={'Full':'full','Single-Agent':'single_agent','Fixed-Pipeline':'fixed_pipeline'}
CAP=600000;WALL=4500;ENVELOPE=16000000;HEADROOM=200000
FIELDS=['case_id','variant','phase','status','attempt','started_utc','finished_utc','wall_seconds','exit_code','reason','pid']


def schedule(public,private):
    rows=[]
    for case in public['cases']:
        spec=private[case['case_id']]['spec']
        order=spec['config_execution_order']
        if len(order)!=3 or set(order)!=set(VARIANTS):raise ValueError('Invalid frozen configuration order')
        for name in order:
            rows.append(dict(case_id=case['case_id'],variant=VARIANTS[name],phase='formal',status='PENDING',attempt=1,
                started_utc='',finished_utc='',wall_seconds='',exit_code='',reason='',pid=''))
    return rows


def consumed(study):
    known=reserved=0
    for attempt in (study/'audits').glob('*/*/*/attempt-*'):
        path=attempt/'EXECUTION_RECEIPT.json'
        if path.exists():
            receipt=json.loads(path.read_text())
            known+=receipt.get('new_tokens',receipt.get('known_tokens',0))
            reserved+=receipt.get('reserved_unknown_tokens',0)
        else:
            # Hard timeouts may precede finally/receipt: retain provider-reported usage.
            for telemetry in attempt.glob('*/model_calls/telemetry.json'):
                known+=json.loads(telemetry.read_text()).get('total_tokens',0) or 0
    return known,reserved


def can_start(now,deadline,known,reserved):
    return (now+timedelta(seconds=WALL)<=deadline-timedelta(hours=12)
            and known+reserved+CAP+HEADROOM<=ENVELOPE)


def failure_kind(output,code):
    if code==124:return 'TIME_LIMIT'
    records=[]
    for path in output.glob('*/model_calls/*.json'):
        if path.name.startswith(('telemetry','runtime_snapshot')):continue
        value=json.loads(path.read_text());records.append(value)
    # A bounded invocation refusal is an observed resource outcome, not a scientific negative.
    if any(r.get('failure_category')=='budget_exceeded' for r in records):return 'RESOURCE_LIMIT'
    return 'ENGINEERING_ERROR'


def frozen_inputs(study):
    public=json.loads((study/'FORMAL_PUBLIC_INPUTS.json').read_text())
    private={case['case_id']:json.loads((study/'private/formal'/(case['case_id']+'.json')).read_text()) for case in public['cases']}
    if len(public['cases'])!=7:raise ValueError('Expected seven cases')
    hashes={}
    for case in public['cases']:
        record=private[case['case_id']]
        for entry in record['records']:
            path=Path(case['training'])/('images/'+entry['img_id']+'.png')
            if sha(path)!=entry['public_file_sha256']:raise ValueError('Public training image changed')
            hashes[str(path)]=entry['public_file_sha256']
        for part in ('training','discovery'):
            path=Path(case[part])/'metadata.csv';hashes[str(path)]=sha(path)
        if hashes[str(Path(case['training'])/'metadata.csv')]!=case['metadata_sha256']:raise ValueError('Metadata changed')
    return public,private,hashes


def freeze(study):
    directory=study/'protocol/formal-main-v1'
    if directory.exists():raise ValueError('Freeze already exists; verify it, never overwrite')
    public,private,hashes=frozen_inputs(study)
    profile=json.loads((study/'protocol/one-construction-alpha-aligned.json').read_text())
    adaptive=profile['execution_options']['adaptive_screening']
    if (profile['promotion_policy']['max_probe_candidates']!=1 or adaptive['candidate_budget']!=1
        or adaptive['family_alpha']!=0.05/3 or adaptive['minimum_effect']!=0.02
        or profile['execution_options']['training_options']['seeds']!=[42]):raise ValueError('Common profile mismatch')
    directory.mkdir(parents=True)
    write(directory/'profile.json',profile)
    rows=schedule(public,private)
    write(directory/'schedule.json',rows)
    write(directory/'input-hashes.json',hashes)
    deadline=json.loads((study/'STUDY_START.json').read_text())['experiment_deadline_utc']
    known,reserved=consumed(study)
    rules={'version':'eswa-formal-main-v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
        'audit_count':len(rows),'tasks':7,'independent_source_datasets':1,
        'per_audit_known_plus_reserved_token_stop':CAP,'per_audit_wall_seconds':WALL,
        'study_planning_envelope':ENVELOPE,'global_start_headroom':HEADROOM,
        'usage_before_formal':{'known':known,'reserved_estimates':reserved,
            'limitations':'Historical missing usage cannot be recovered as exact billing; estimates and actual known usage stay separate.'},
        'worst_main_known_planning_tokens':known+len(rows)*CAP,
        'deadline_utc':deadline,'final_export_hours':12,
        'shared_options':{'compact_role_protocol':'eswa-compact-role-contract-v1.1','qualitative_samples':12,
            'qualitative_tasks':1,'construction_candidates':1,'max_discovery_candidates':5,
            'per_probe_alpha':0.05/3,'minimum_effect':0.02,'training_seed':42,
            'cross_run_learning':False,'retrieval':False},
        'completion_rules':{'small_or_negative_effect':'retain_without_rerun','full_not_best':'retain_without_rerun',
            'resource_limit':'retain_failed_audit_and_partial_artifacts_in_denominator',
            'engineering_error':'stop_queue; repair; freeze a new version uniformly for affected complete task blocks, retaining originals',
            'human_adjudication':'PENDING until actual independent human labels exist; never infer or fabricate human review',
            'discovery_scoring':'Keep E1 known-cue rediscovery separate from E2 blind discovery; autonomous candidate acceptance is not oracle validation.',
            'oracle_operations':'Evaluator-only diagnostics labelled separately; never passed to Agents',
            'dependence':'Seven conditions share sources and split; no claim of seven independent datasets.',
            'replication':'Optional seeds43/44 are post-audit frozen-intervention classifier checks, not independent Agent repetitions.'},
        'extensions':'Ablations/Pet/replication require a separately frozen feasible whole-block plan within remaining budget/time; no automatic expansion.'}
    rules['engineering_readiness']={'cached_training_publication_replay':'success; zero new API calls/training runs',
        'completed_development_agent_audits':0,'limitation':'End-to-end completion under the revised 600k protocol has not been demonstrated. Resource failures are valid feasibility outcomes, not empirical absence of shortcuts.',
        'training_wall_seconds_observed':395.48399999999856,
        'gpu_time_definition':'Profile elapsed_seconds covers GPU training plus validation/checkpoints/host work; it is not measured GPU kernel active time. Report separately from engineering replay wall time.'}
    write(directory/'rules.json',rules)
    identity=source_identity(require_clean=False)
    archive=archive_source(directory/'source.zip',identity['source_tree_sha256'])
    write(directory/'FREEZE.json',{'source_tree_sha256':identity['source_tree_sha256'],'source_archive_sha256':archive,
        'profile_sha256':sha(directory/'profile.json'),'schedule_sha256':sha(directory/'schedule.json'),
        'rules_sha256':sha(directory/'rules.json'),'input_hashes_sha256':sha(directory/'input-hashes.json'),
        'public_index_sha256':sha(study/'FORMAL_PUBLIC_INPUTS.json'),'frozen_before_first_formal_call':True})
    print(json.dumps({'status':'FROZEN','audits':len(rows),'known_before_formal':known,'main_planning_tokens':known+len(rows)*CAP}))


def verify(study):
    directory=study/'protocol/formal-main-v1';frozen=json.loads((directory/'FREEZE.json').read_text())
    for filename,key in [('source.zip','source_archive_sha256'),('profile.json','profile_sha256'),
        ('schedule.json','schedule_sha256'),('rules.json','rules_sha256'),('input-hashes.json','input_hashes_sha256')]:
        if sha(directory/filename)!=frozen[key]:raise ValueError('Frozen artifact changed: '+filename)
    if source_identity(require_clean=False)['source_tree_sha256']!=frozen['source_tree_sha256']:raise ValueError('Source differs from frozen formal version')
    if sha(study/'FORMAL_PUBLIC_INPUTS.json')!=frozen['public_index_sha256']:raise ValueError('Public index changed')
    for path,digest in json.loads((directory/'input-hashes.json').read_text()).items():
        if sha(Path(path))!=digest:raise ValueError('Frozen input changed: '+path)
    return directory


def run(study):
    lock=(study/'formal.queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('A formal queue is already active')
    directory=verify(study)
    status=study/'FORMAL_TASK_STATUS.csv'
    if status.exists():
        with status.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    else:rows=json.loads((directory/'schedule.json').read_text())
    deadline=datetime.fromisoformat(json.loads((directory/'rules.json').read_text())['deadline_utc'])
    def persist(stage):
        temp=status.with_suffix('.csv.tmp')
        with temp.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
        temp.replace(status)
        write(study/'LIVE_STATUS.json',{'stage':stage,'queue_pid':os.getpid(),
            'formal_started':any(r['started_utc'] for r in rows),'tasks':rows,
            'updated_utc':datetime.now(timezone.utc).isoformat(),'next_gate':'Retain outcomes; stop on engineering error. Review complete main blocks before any extensions.'})
    for row in rows:
        output=study/'audits/formal'/row['case_id']/row['variant']/'attempt-01'
        if row['status']=='RUNNING' or (row['status']=='PENDING' and output.exists()):
            if alive(row.get('pid')):raise SystemExit('Recorded worker remains alive; do not duplicate')
            persist('formal_interruption_requires_repair');return
        if row['status']=='FAILED':persist('formal_engineering_failure_requires_repair');return
        if row['status']!='PENDING':continue
        known,reserved=consumed(study)
        if not can_start(datetime.now(timezone.utc),deadline,known,reserved):
            row['reason']='Not started: remaining whole-audit budget/time guard'
            persist('formal_scope_stop_requires_export');return
        if source_identity(require_clean=False)['source_tree_sha256']!=json.loads((directory/'FREEZE.json').read_text())['source_tree_sha256']:
            persist('formal_source_change_requires_repair');return
        command=[sys.executable,'-X','utf8',str(ROOT/'experiments/eswa_20261019/run_one.py'),
            '--study-root',str(study),'--case-id',row['case_id'],'--variant',row['variant'],
            '--phase','formal','--profile',str(directory/'profile.json'),'--token-budget',str(CAP),
            '--attempt','1','--compact-role-prompts','--reserve-missing-text-usage',
            '--qualitative-sample-cap','12','--qualitative-task-cap','1']
        name=row['case_id']+'-'+row['variant']
        begin=time.monotonic()
        with (study/f'logs/formal-{name}.stdout.log').open('xb') as out,(study/f'logs/formal-{name}.stderr.log').open('xb') as err:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            row.update(status='RUNNING',pid=process.pid,started_utc=datetime.now(timezone.utc).isoformat())
            persist('formal_main_running')
            try:code=process.wait(timeout=WALL)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                process.wait();code=124
        kind='COMPLETED' if code==0 else failure_kind(output,code)
        row.update(status='COMPLETED' if code==0 else ('FAILED' if kind=='ENGINEERING_ERROR' else 'FAILED_RETAINED'),
            exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=round(time.monotonic()-begin,2),reason=kind)
        if kind=='ENGINEERING_ERROR':persist('formal_engineering_failure_requires_repair');return
        persist('formal_main_running')
    persist('formal_main_complete_requires_extensions_review')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','verify','run'])
    action=parser.parse_args().action;study=ROOT/'results_eswa_single'
    if action=='freeze':freeze(study)
    elif action=='verify':verify(study);print('Frozen source, schedule, rules and inputs verified')
    else:run(study)


if __name__=='__main__':main()
