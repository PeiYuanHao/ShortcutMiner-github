"""One bounded cached-phase development validation of the shared one-construction policy."""
import csv
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import msvcrt

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repair-alpha-contract',action='store_true')
    repair=parser.parse_args().repair_alpha_contract
    study=ROOT/'results_eswa_single';lock=(study/'development.queue.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Another queue is active')
    original_marker=study/'RESOURCE_PROTOCOL_VALIDATION.json'
    marker=study/('RESOURCE_ALPHA_REPAIR.json' if repair else 'RESOURCE_PROTOCOL_VALIDATION.json')
    if marker.exists():raise SystemExit('This bounded validation has already been scheduled')
    status=study/'DEVELOPMENT_TASK_STATUS.csv'
    with status.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    row=next(r for r in rows if r['case_id']=='dev-03')
    if row['status']!='FAILED':raise SystemExit('Expected retained failed resource-heavy validation')
    prior=study/('audits/development/dev-03/single_agent/attempt-03' if repair else 'audits/development/dev-03/single_agent/attempt-02')
    receipt=json.loads((prior/'EXECUTION_RECEIPT.json').read_text())
    profile=json.loads((study/'protocol/development-profile.json').read_text(encoding='utf-8-sig'))
    if repair:
        from experiments.eswa_20261019.resource_policy import single_probe_profile
        profile=single_probe_profile(profile)
        cap=json.loads(original_marker.read_text())['cumulative_development_stop']
        if int(row['attempt'])!=3 or cap<=receipt['known_tokens']:raise SystemExit('Unexpected repair state or exhausted original quota')
    else:
        profile['promotion_policy']['max_probe_candidates']=1
        cap=receipt['known_tokens']+200000
    profile_path=study/('protocol/one-construction-alpha-aligned.json' if repair else 'protocol/one-construction-profile.json')
    write(profile_path,profile)
    attempt=4 if repair else 3
    log_name='dev-03-resource-alpha-repair' if repair else 'dev-03-resource-protocol'
    write(marker,{'previous_status':rows,'protocol':'one-construction-one-qualitative-task-v1',
        'incremental_development_token_cap':cap-receipt['known_tokens'],'prior_cumulative_known_tokens':receipt['known_tokens'],
        'cumulative_development_stop':cap,'formal_token_stop_proposed':600000,
        'common_limits':{'max_candidates':5,'construction_candidates':1,'qualitative_tasks':1,'qualitative_samples':12},
        'scope':'Development-only fork of cached training phases, not formal evidence. Original failures and full controller memory retained.',
        'selection':'First executable candidate in frozen order; no hidden truth or model outcome used.',
        'formal_scope':'7 tasks x 3 configurations; source/protocol not yet frozen.'})
    def persist(stage):
        temp=status.with_suffix('.csv.tmp')
        with temp.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        temp.replace(status)
        write(study/'LIVE_STATUS.json',{'stage':stage,'queue_pid':os.getpid(),'formal_started':False,
            'updated_utc':datetime.now(timezone.utc).isoformat(),'tasks':rows,
            'next_gate':'Check the resource-bounded workflow; preserve failures. Freeze and implement formal queue only after readiness review.'})
    command=[sys.executable,'-X','utf8',str(ROOT/'experiments/eswa_20261019/run_one.py'),
        '--study-root',str(study),'--case-id','dev-03','--variant','single_agent','--phase','development',
        '--profile',str(profile_path),'--token-budget',str(cap),
        '--attempt',str(attempt),'--compact-role-prompts','--reserve-missing-text-usage','--qualitative-sample-cap','12',
        '--qualitative-task-cap','1','--resource-policy-development-fork','--continue-development-from',str(prior)]
    begin=time.time()
    with (study/f'logs/{log_name}.stdout.log').open('xb') as out, (study/f'logs/{log_name}.stderr.log').open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        row.update(status='RUNNING',attempt=attempt,pid=process.pid,started_utc=datetime.now(timezone.utc).isoformat(),
            finished_utc='',wall_seconds='',exit_code='',reason='Cached-phase validation of one-construction/one-visual-task shared resource protocol.')
        persist('development_resource_protocol_running')
        try:code=process.wait(timeout=4500)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            process.wait();code=124
    row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,
        finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=round(time.time()-begin,2))
    persist('development_resource_protocol_complete_requires_freeze' if code==0 else 'development_gate_requires_repair')


if __name__=='__main__':main()
