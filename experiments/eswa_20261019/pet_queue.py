"""One Pet audit; retain original failures and advance to offline diagnostics."""
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
from experiments.eswa_20261019.extension_queue import costs,classify
from shortcutminer.experiment.waterbirds_paper import sha,source_identity

MAIN=ROOT/'results_eswa_single';STUDY=MAIN/'extensions/pet-v1'


def main():
    lock=(MAIN/'extension.queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Another extension queue is active')
    if (STUDY/'TASK_STATUS.csv').exists():raise ValueError('Pet already scheduled; inspect outcome, never duplicate')
    protocol=STUDY/'protocol';frozen=json.loads((protocol/'FREEZE.json').read_text())
    if source_identity(require_clean=False)['source_tree_sha256']!=frozen['source_tree_sha256']:raise ValueError('Pet source changed')
    for name,digest in frozen['files'].items():
        if sha(protocol/name)!=digest:raise ValueError('Pet freeze changed')
    if sha(STUDY/'FORMAL_PUBLIC_INPUTS.json')!=frozen['public_index_sha256']:raise ValueError('Pet index changed')
    for path,digest in json.loads((protocol/'input-hashes.json').read_text()).items():
        if sha(Path(path))!=digest:raise ValueError('Pet public input changed')
    known,reserved=costs();deadline=datetime.fromisoformat(json.loads((MAIN/'STUDY_START.json').read_text())['experiment_deadline_utc'])
    now=datetime.now(timezone.utc)
    if known+reserved+600000+200000>16000000 or now+timedelta(seconds=4500)>deadline-timedelta(hours=12) or now>deadline-timedelta(hours=24):
        raise ValueError('Pet whole-audit budget/time guard')
    row={'case_id':'r-01','variant':'full','phase':'formal','status':'RUNNING','attempt':1,
        'study_root':str(STUDY),'started_utc':now.isoformat(),'finished_utc':'','wall_seconds':'','exit_code':'','reason':'','pid':''}
    def persist(stage):
        with (STUDY/'TASK_STATUS.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        write(MAIN/'LIVE_STATUS.json',{'stage':stage,'queue_pid':os.getpid(),'formal_started':True,'tasks':[row],
            'updated_utc':datetime.now(timezone.utc).isoformat(),'next_gate':'Retain real-case outcome/partial artifacts, then evaluator-only local background/no-op/placebo diagnostics and source-grounded analysis/export. No automatic audit retry.'})
    begin=time.monotonic()
    with (STUDY/'logs/audit.stdout.log').open('xb') as out,(STUDY/'logs/audit.stderr.log').open('xb') as err:
        process=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'experiments/eswa_20261019/run_pet.py')],
            cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        row['pid']=process.pid;persist('pet_audit_running')
        try:code=process.wait(timeout=4500)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            process.wait();code=124
    output=STUDY/'audits/formal/r-01/full/attempt-01'
    error=(STUDY/'logs/audit.stderr.log').read_text(encoding='utf-8',errors='replace')[-6000:]
    kind='COMPLETED' if code==0 else classify(output,code,error)
    row.update(status='COMPLETED' if code==0 else ('FAILED' if kind=='ENGINEERING_ERROR' else 'FAILED_RETAINED'),
        finished_utc=datetime.now(timezone.utc).isoformat(),exit_code=code,reason=kind,wall_seconds=round(time.monotonic()-begin,2))
    persist('pet_engineering_failure_requires_repair' if kind=='ENGINEERING_ERROR' else 'pet_audit_ended_requires_local_diagnostics_and_export')


if __name__=='__main__':main()
