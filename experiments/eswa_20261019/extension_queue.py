"""Bounded six-arm controlled extension, on a separately frozen version."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import json
import msvcrt
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.e4_ablation.variants.policies import write
from experiments.eswa_20261019.formal_queue import consumed,failure_kind
from experiments.eswa_20261019.monitor import alive
from shortcutminer.experiment.waterbirds_paper import sha,source_identity,archive_source

MAIN=ROOT/'results_eswa_single';STUDY=MAIN/'extensions/fixed-candidate-v1'
CONDITIONS=['full_same_candidates','without_natural_language_semantic_feedback','fixed_routing']
FIELDS=['case_id','variant','phase','status','attempt','condition','study_root','started_utc','finished_utc','wall_seconds','exit_code','reason','pid']


def costs():
    known,reserved=consumed(MAIN)
    for root in (MAIN/'extensions').iterdir():
        if root.is_dir():
            k,r=consumed(root);known+=k;reserved+=r
    return known,reserved


def freeze():
    if STUDY.exists():raise ValueError('Extension study already exists; inspect, never overwrite')
    plan=json.loads((MAIN/'extensions/SCOPE_PLAN.json').read_text())
    public=json.loads((MAIN/'FORMAL_PUBLIC_INPUTS.json').read_text())['cases']
    profile=MAIN/'protocol/formal-main-v1/profile.json'
    arms=[];cases=[];rng=random.Random(2026102001)
    for block,selected in enumerate(plan['ablation_task_blocks']):
        case=next(row for row in public if row['case_id']==selected['case_id'])
        source=Path(selected['discovery_source']);measure=source.parent/(case['case_id']+'-full-construction')/'candidate-measurements.json'
        source_record=json.loads((source/'source_identity.json').read_text())
        if json.loads((source/'run.json').read_text())['status']!='COMPLETED':raise ValueError('Source discovery incomplete')
        if sha(source/'source_snapshot.zip')!=source_record['archive_sha256']:raise ValueError('Source archive changed')
        paths=[source/'identify/output.json',source/'identify/raw_output.json',source/'frozen_execution_config.json',
            source/'source_snapshot.zip',source/'source_identity.json',measure]
        paths.extend(sorted((source/'qualitative').glob('*/result.json')))
        hashes={str(path):sha(path) for path in paths}
        order=list(CONDITIONS);rng.shuffle(order)
        for number,condition in enumerate(order):
            cid=f'a-{block+1:02d}-{number+1:02d}'
            arms.append({'case_id':cid,'condition':condition,'source_case_id':case['case_id'],
                'discovery_source':str(source),'measurement_source':str(measure),'frozen_phase_hashes':hashes})
            cases.append({**case,'case_id':cid})
    known,reserved=costs()
    if known+reserved+6*600000+600000+200000>16000000:raise ValueError('Cannot fit full ablation blocks plus reserved Pet')
    STUDY.mkdir(parents=True);protocol=STUDY/'protocol';protocol.mkdir();(STUDY/'logs').mkdir()
    shutil.copyfile(profile,protocol/'profile.json')
    write(STUDY/'FORMAL_PUBLIC_INPUTS.json',{'cases':cases,'scope':'Fixed-candidate conditional extension, not new blind discovery.'})
    write(protocol/'CONTROLLED_INPUTS.json',{'arms':arms,'discovery_calls_allowed':False,'measurement_plan_calls_allowed':False})
    rows=[{**dict.fromkeys(FIELDS,''),'case_id':arm['case_id'],'variant':'full','phase':'formal','status':'PENDING',
        'attempt':1,'condition':arm['condition'],'study_root':str(STUDY)} for arm in arms]
    write(protocol/'schedule.json',rows)
    write(protocol/'RULES.json',{'scope_plan_sha256':sha(MAIN/'extensions/SCOPE_PLAN.json'),
        'candidate_entry':'Exactly the same training-only discovery, observations, candidate ordering and primary measurements in each three-arm block. Construction/validation rerun for all arms.',
        'feedback_mask_version':'construction-and-qualitative-text-feedback-v1',
        'mask_limitations':'Only the exact free-text fields under selected qualitative/visual/Executive feedback contexts are withheld. Categoricals, identities, numbers, tool permissions and raw canonical reviews remain; this is not removal of the visual branch or all semantic information.',
        'common_caps':{'known_plus_reserved_tokens':600000,'wall_seconds':4500,'construction_candidates':1,'qualitative_tasks':1,'qualitative_samples':12},
        'pet_reserved_tokens':600000,'global_headroom':200000,'main_immutable':True,
        'source_dependence':'Same previously used source pool, candidates and held-out records: conditional comparative diagnostics, not new independent confirmation.',
        'failures':'Resource, provider and model protocol failures retained in denominator; code/backend failures stop queue. No retries for effect size or Full not winning.',
        'human_adjudication':'PENDING until real humans supply labels.',
        'deadline_utc':json.loads((MAIN/'STUDY_START.json').read_text())['experiment_deadline_utc']})
    identity=source_identity(require_clean=False)
    archive=archive_source(protocol/'source.zip',identity['source_tree_sha256'])
    files=['profile.json','CONTROLLED_INPUTS.json','schedule.json','RULES.json','source.zip']
    write(protocol/'FREEZE.json',{'source_tree_sha256':identity['source_tree_sha256'],
        'source_archive_sha256':archive,'files':{name:sha(protocol/name) for name in files},
        'public_index_sha256':sha(STUDY/'FORMAL_PUBLIC_INPUTS.json'),
        'main_input_hashes_sha256':sha(MAIN/'protocol/formal-main-v1/input-hashes.json')})
    print(json.dumps({'status':'FROZEN','controlled_audits':6,'known_before_extensions':known,'reserved_estimates':reserved}))


def verify():
    protocol=STUDY/'protocol';frozen=json.loads((protocol/'FREEZE.json').read_text())
    if source_identity(require_clean=False)['source_tree_sha256']!=frozen['source_tree_sha256']:raise ValueError('Extension source changed')
    for name,digest in frozen['files'].items():
        if sha(protocol/name)!=digest:raise ValueError('Extension frozen artifact changed')
    if sha(STUDY/'FORMAL_PUBLIC_INPUTS.json')!=frozen['public_index_sha256']:raise ValueError('Extension public index changed')
    manifest=json.loads((protocol/'CONTROLLED_INPUTS.json').read_text())
    for arm in manifest['arms']:
        for path,digest in arm['frozen_phase_hashes'].items():
            if sha(Path(path))!=digest:raise ValueError('Frozen controlled discovery changed')
    # The main data manifest predates outcomes; verify bytes without rerunning main audits.
    path=MAIN/'protocol/formal-main-v1/input-hashes.json'
    if sha(path)!=frozen['main_input_hashes_sha256']:raise ValueError('Main data manifest changed')
    for path,digest in json.loads(path.read_text()).items():
        if sha(Path(path))!=digest:raise ValueError('Shared public input changed')
    return frozen


def classify(output,code,error_text=''):
    kind=failure_kind(output,code)
    if kind!='ENGINEERING_ERROR':return kind
    if 'mandatory_quantitative_backend_execution_failed' in error_text:return 'ENGINEERING_ERROR'
    records=[json.loads(path.read_text()) for path in output.glob('*/model_calls/*.json')
        if not path.name.startswith(('telemetry','runtime_snapshot'))]
    failures=[record for record in records if record.get('failure_category')]
    if failures:
        last=max(failures,key=lambda record:record.get('completed_at',''))
        if last['failure_category'] in {'protocol_schema_failed','invalid_json'} and any(message in error_text for message in (
            'Model JSON did not satisfy the required protocol schema','Model response content is not valid JSON')):return 'MODEL_PROTOCOL_FAILURE'
        if last['failure_category']=='invalid_response_envelope' and 'Model endpoint returned an invalid JSON envelope' in error_text:return 'PROVIDER_FAILURE'
    return 'ENGINEERING_ERROR'


def run():
    lock=(MAIN/'extension.queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Extension queue already active')
    frozen=verify();protocol=STUDY/'protocol';status=STUDY/'TASK_STATUS.csv'
    if status.exists():
        with status.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    else:rows=json.loads((protocol/'schedule.json').read_text())
    deadline=datetime.fromisoformat(json.loads((protocol/'RULES.json').read_text())['deadline_utc'])
    def persist(stage):
        temp=status.with_suffix('.tmp')
        with temp.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
        temp.replace(status)
        write(MAIN/'LIVE_STATUS.json',{'stage':stage,'queue_pid':os.getpid(),'formal_started':True,'tasks':rows,
            'updated_utc':datetime.now(timezone.utc).isoformat(),
            'next_gate':'Complete same-source controlled blocks with failures retained; Pet harness and diagnostics still require implementation.'})
    for row in rows:
        output=STUDY/'audits/formal'/row['case_id']/'full/attempt-01'
        if row['status']=='RUNNING' or (row['status']=='PENDING' and output.exists()):
            if alive(row.get('pid')):raise SystemExit('Worker alive; do not duplicate')
            persist('extension_interruption_requires_repair');return
        if row['status']=='FAILED':persist('extension_engineering_failure_requires_repair');return
        if row['status']!='PENDING':continue
        known,reserved=costs();now=datetime.now(timezone.utc)
        if known+reserved+600000+600000+200000>16000000 or now+timedelta(seconds=4500)>deadline-timedelta(hours=12) or now>deadline-timedelta(hours=24):
            row['reason']='Whole-audit budget/time guard; Pet allowance retained';persist('extension_scope_stop_requires_review');return
        if source_identity(require_clean=False)['source_tree_sha256']!=frozen['source_tree_sha256']:
            persist('extension_source_change_requires_repair');return
        command=[sys.executable,'-X','utf8',str(ROOT/'experiments/eswa_20261019/run_extension.py'),
            '--study-root',str(STUDY),'--case-id',row['case_id']]
        begin=time.monotonic()
        with (STUDY/'logs'/(row['case_id']+'.stdout.log')).open('xb') as out,(STUDY/'logs'/(row['case_id']+'.stderr.log')).open('xb') as err:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            row.update(status='RUNNING',pid=process.pid,started_utc=now.isoformat());persist('controlled_extensions_running')
            try:code=process.wait(timeout=4500)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                process.wait();code=124
        error_text=(STUDY/'logs'/(row['case_id']+'.stderr.log')).read_text(encoding='utf-8',errors='replace')[-6000:]
        kind='COMPLETED' if code==0 else classify(output,code,error_text)
        row.update(status='COMPLETED' if code==0 else ('FAILED' if kind=='ENGINEERING_ERROR' else 'FAILED_RETAINED'),
            finished_utc=datetime.now(timezone.utc).isoformat(),exit_code=code,
            wall_seconds=round(time.monotonic()-begin,2),reason=kind)
        if kind=='ENGINEERING_ERROR':persist('extension_engineering_failure_requires_repair');return
        persist('controlled_extensions_running')
    persist('controlled_extensions_complete_requires_pet_and_analysis')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','verify','run']);action=parser.parse_args().action
    if action=='freeze':freeze()
    elif action=='verify':verify();print('Extension source, controlled inputs and common public data verified')
    else:run()


if __name__=='__main__':main()
