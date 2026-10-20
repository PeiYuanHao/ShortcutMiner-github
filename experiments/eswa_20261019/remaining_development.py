"""Run the last untouched development input after bounded-usage repairs."""
import csv
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
    study=ROOT/'results_eswa_single';lock=(study/'development.queue.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Existing development queue is active')
    status=study/'DEVELOPMENT_TASK_STATUS.csv'
    with status.open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    row=next(r for r in rows if r['case_id']=='dev-03')
    if row['status']!='PENDING':raise SystemExit('Last development input has already been attempted')
    if (study/'BOUNDED_USAGE_DEVELOPMENT.json').exists():raise SystemExit('This validation was already scheduled')
    write(study/'BOUNDED_USAGE_DEVELOPMENT.json',{'previous_status':rows,
        'root_cause':'A timed-out text request lacked billing usage; the old shared budget stopped all further calls.',
        'repairs':['Reserve conservative text usage for unreported failures; never report it as known or zero.',
                   'Keep unknown multimodal usage fail-closed. Preserve retry/call/token ceilings.',
                   'Bound ordinary qualitative evidence to 12 balanced samples rather than 40; report changed coverage.'],
        'target':'Untouched dev-03 Single-Agent, 300000 tokens; no further dev-02 rediscovery or quota increase.'})
    for old in rows:
        if old['status']=='FAILED':old.update(status='FAILED_RETAINED',reason='Text timeout and incomplete billing telemetry retained; no scientific negative inferred.')
    def persist(stage):
        temp=status.with_suffix('.csv.tmp')
        with temp.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        temp.replace(status)
        write(study/'LIVE_STATUS.json',{'stage':stage,'queue_pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),'formal_started':False,'tasks':rows,
            'next_gate':'Inspect bounded-context/usage validation and training throughput, then freeze the formal protocol.'})
    command=[sys.executable,'-X','utf8',str(ROOT/'experiments/eswa_20261019/run_one.py'),
        '--study-root',str(study),'--case-id','dev-03','--variant','single_agent','--phase','development',
        '--profile',str(study/'protocol/development-profile.json'),'--token-budget','300000',
        '--attempt','1','--compact-role-prompts','--reserve-missing-text-usage','--qualitative-sample-cap','12']
    begin=time.time()
    with (study/'logs/dev-03-single_agent-bounded-usage.stdout.log').open('xb') as out, (study/'logs/dev-03-single_agent-bounded-usage.stderr.log').open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        row.update(status='RUNNING',attempt=1,pid=process.pid,started_utc=datetime.now(timezone.utc).isoformat(),
            finished_utc='',wall_seconds='',exit_code='',reason='Bounded usage and 12-sample qualitative development protocol.')
        persist('development_audit_running')
        try:code=process.wait(timeout=4500)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            process.wait();code=124
    row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,
        finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=round(time.time()-begin,2))
    persist('development_complete_requires_formal_review' if code==0 else 'development_gate_requires_repair')


if __name__=='__main__':main()
