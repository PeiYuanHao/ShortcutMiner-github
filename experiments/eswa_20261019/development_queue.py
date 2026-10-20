"""Run at most three development audits; stop before formal scientific collection."""
from __future__ import annotations
import csv
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-quota-repair',action='store_true')
    parser.add_argument('--bounded-context-validation',action='store_true',
        help='Use the two untouched development inputs for the new compact protocol; preserve old failures.')
    parser.add_argument('--resume-launch-fix',action='store_true')
    parser.add_argument('--resume-route-contract-fix',action='store_true')
    parser.add_argument('--resume-measured-feasibility',action='store_true')
    args=parser.parse_args()
    study=ROOT/'results_eswa_single'
    lock_path=study/'development.queue.lock'
    import msvcrt
    lock=lock_path.open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Another development queue is already active')
    started=json.loads((study/'STUDY_START.json').read_text())
    deadline=datetime.fromisoformat(started['experiment_deadline_utc']).timestamp()
    rows=[{'case_id':'dev-01','variant':'full','phase':'development','status':'PENDING'},
          {'case_id':'dev-02','variant':'single_agent','phase':'development','status':'PENDING'},
          {'case_id':'dev-03','variant':'fixed_pipeline','phase':'development','status':'PENDING'}]
    statusfile=study/'DEVELOPMENT_TASK_STATUS.csv'
    continuation=None
    if statusfile.exists():
        with statusfile.open(newline='',encoding='utf-8') as stream:
            previous=list(csv.DictReader(stream))
        if any(r['status']!='PENDING' for r in previous) and not (args.resume_quota_repair or args.bounded_context_validation or args.resume_launch_fix or args.resume_route_contract_fix or args.resume_measured_feasibility):
            raise SystemExit('Existing attempts preserved. Inspect results before scheduling another development audit.')
        if args.resume_quota_repair:
            rows=previous
            first=rows[0]
            if first['status']!='FAILED':raise SystemExit('This bounded repair is only for the first failed quota attempt')
            continuation=study/'audits/development/dev-01/full/attempt-01'
            receipt=json.loads((continuation/'EXECUTION_RECEIPT.json').read_text())
            if receipt['known_tokens']<300000 or (study/'DEVELOPMENT_QUOTA_REPAIR.json').exists():
                raise SystemExit('Quota repair unavailable or already attempted; inspect rather than retrying again')
            write(study/'DEVELOPMENT_QUOTA_REPAIR.json',{'previous_status':previous,
                'previous_known_tokens':receipt['known_tokens'],'new_cumulative_development_cap':600000,
                'formal_budget_not_yet_frozen':True,
                'scope':'Reuse completed pre-model phases rather than rerun discovery. Retain every failed attempt and its cost. One bounded development quota repair only.'})
            first.update(status='PENDING',attempt=2,reason='Bounded development quota repair; original attempt preserved.')
        if args.bounded_context_validation:
            if (study/'BOUNDED_CONTEXT_DEVELOPMENT.json').exists():
                raise SystemExit('Bounded-context validation already scheduled; do not duplicate it')
            if previous[0]['status']!='FAILED' or any(r['status']!='PENDING' for r in previous[1:]):
                raise SystemExit('Only the two untouched pending development inputs may be used')
            write(study/'BOUNDED_CONTEXT_DEVELOPMENT.json',{'previous_status':previous,
                'new_role_protocol':'eswa-compact-role-contract-v1','new_audit_cap':300000,
                'selection_basis':'Pre-model API/quota feasibility; no heldout model outcomes have been inspected.',
                'scope':'Dev-02 Full and Dev-03 Single-Agent live; Fixed-Pipeline typed adapters covered by prior offline checks. Original Full attempts remain failed, with all cost retained.',
                'formal_collection_requires_review':True})
            rows=previous
            rows[0].update(status='FAILED_RETAINED',reason='Verbose-prompt quota failures retained; no further retry under that protocol.')
            rows[1].update(variant='full',reason='Untouched input; new compact role protocol development validation.')
            rows[2].update(variant='single_agent',reason='Untouched input; new compact role protocol development validation.')
        if args.resume_launch_fix:
            marker=study/'DEVELOPMENT_LAUNCH_FIX.json'
            failed=previous[1]
            no_run=not (study/'audits/development'/failed['case_id']/failed['variant']).exists()
            if marker.exists() or failed['status']!='FAILED' or str(failed['exit_code'])!='2' or not no_run:
                raise SystemExit('Only a pre-provider argument parsing failure may be repaired here')
            write(marker,{'previous_status':previous,'repair':'Normalize empty CSV attempt to integer 1.',
                'provider_calls_in_failed_launch':0,'scientific_results_in_failed_launch':False})
            rows=previous;rows[1].update(status='PENDING',reason='Launch-only CSV integer repair; no provider call was made.')
            args.bounded_context_validation=True
        if args.resume_measured_feasibility:
            marker=study/'MEASURED_BUDGET_FEASIBILITY.json'
            continuation=study/'audits/development/dev-03/single_agent/attempt-01'
            receipt=json.loads((continuation/'EXECUTION_RECEIPT.json').read_text())
            failed=previous[2]
            if marker.exists() or failed['status']!='FAILED' or receipt['known_tokens']<300000 or receipt['usage_incomplete']:
                raise SystemExit('Measured feasibility continuation unavailable or already attempted')
            write(marker,{'previous_status':previous,'observed_pre_validation_known_tokens':receipt['known_tokens'],
                'cumulative_single_agent_development_cap':600000,
                'justification':'Three distinct development audits establish 300k infeasibility; use the existing predeclared 600k alternative rather than repeat discovery or lower scientific gates.',
                'formal_scope_decision':'Before any formal outcomes, remove all three rho=.75 tasks: 7 tasks x 3 configurations. Keep known+reserved usage within the existing ~16M study envelope; remaining extensions conditional on budget.',
                'common_formal_options_proposed':{'token_stop':600000,'qualitative_samples':12,'role_protocol':'eswa-compact-role-contract-v1.1'},
                'reuse':'Completed training-only discovery, construction receipts and the same persistent Single-Agent memory; no model outcome existed.'})
            rows=previous;rows[2].update(status='PENDING',attempt=2,reason='Measured 600k alternative; restore existing Single-Agent memory and completed phases.')
            args.bounded_context_validation=True
        if args.resume_route_contract_fix:
            marker=study/'DEVELOPMENT_ROUTE_CONTRACT_FIX.json'
            failed=previous[1];continuation=study/'audits/development/dev-02/full/attempt-01'
            if marker.exists() or failed['status']!='FAILED' or str(failed['exit_code'])!='1':
                raise SystemExit('Routing repair was already attempted or is not applicable')
            write(marker,{'previous_status':previous,
                'repair':'Explicit route/branch/missing-requirements invariants in compact Executive prompt; retain rejected drafts.',
                'cumulative_development_case_cap':600000,'formal_budget_not_frozen':True,
                'reused':'Completed discovery, measurements and training-only constructions; no task model/outcome existed.'})
            rows=previous;rows[1].update(status='PENDING',attempt=2,reason='Executive route-contract repair; reuse pre-model phases.')
            args.bounded_context_validation=True
    fields=['case_id','variant','phase','status','attempt','started_utc','finished_utc','wall_seconds','exit_code','reason','pid']
    def persist(stage):
        with statusfile.with_suffix('.csv.tmp').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        statusfile.with_suffix('.csv.tmp').replace(statusfile)
        write(study/'LIVE_STATUS.json',{'stage':stage,'queue_pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),'formal_started':False,
            'tasks':rows,'next_gate':'Inspect throughput, token budget and construction failures before freezing formal protocol.'})
    persist('development_queue_starting')
    for row in rows:
        if row['status'] not in {'PENDING'}:continue
        row['attempt']=int(row.get('attempt') or 1)
        if time.time()+4500>deadline-12*3600:
            row.update(status='NOT_STARTED',reason='Deadline reserve');persist('deadline_stop');break
        logs=study/'logs';logs.mkdir(exist_ok=True)
        suffix=f"-attempt{row['attempt']}" if args.resume_quota_repair else ''
        if args.resume_launch_fix:suffix='-launchfix1'
        if args.resume_route_contract_fix:suffix='-routefix1'
        if args.resume_measured_feasibility:suffix='-feasibility1'
        stdout=logs/(row['case_id']+'-'+row['variant']+suffix+'.stdout.log')
        stderr=logs/(row['case_id']+'-'+row['variant']+suffix+'.stderr.log')
        command=[sys.executable,'-X','utf8',str(ROOT/'experiments/eswa_20261019/run_one.py'),
            '--study-root',str(study),'--case-id',row['case_id'],'--variant',row['variant'],
            '--phase','development','--profile',str(study/'protocol/development-profile.json'),
            '--token-budget','600000' if args.resume_quota_repair or args.resume_measured_feasibility or (args.resume_route_contract_fix and row['case_id']=='dev-02') else '300000',
            '--attempt',str(row['attempt'])]
        if continuation is not None and row['case_id']==json.loads((continuation/'EXECUTION_RECEIPT.json').read_text())['case_id']:
            command+=['--continue-development-from',str(continuation)]
        if args.bounded_context_validation:command+=['--compact-role-prompts']
        if args.resume_measured_feasibility:command+=['--reserve-missing-text-usage','--qualitative-sample-cap','12']
        begin=time.time()
        with stdout.open('xb') as out,stderr.open('xb') as err:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW)
            row.update(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),pid=process.pid,
                finished_utc='',wall_seconds='',exit_code='')
            persist('development_audit_running')
            try: code=process.wait(timeout=4500)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                process.wait();code=124
        row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,
            finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=round(time.time()-begin,2),
            reason='' if code==0 else 'Inspect retained attempt; no automatic scientific-result retries.')
        persist('development_audit_finished')
        if code!=0:
            persist('development_gate_requires_repair');return
    persist('bounded_context_validation_complete_requires_review' if args.bounded_context_validation else 'development_complete_requires_formal_freeze')


if __name__=='__main__':main()
