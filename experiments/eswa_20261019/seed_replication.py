"""Predeclared classifier-only extra seeds; honor original frozen probe expansion gates."""
import argparse,csv,json,os,subprocess,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from PIL import Image
from pydantic import TypeAdapter
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.e4_ablation.variants.policies import write
from experiments.eswa_20261019.run_one import cache_identity
from shortcutminer.experiment.waterbirds_paper import sha,initialization,source_identity,archive_source
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.intervention.engine import InterventionEngine
from shortcutminer.intervention.protocol import ImageInterventionSpec
from shortcutminer.task_models.torchvision_matrix import TorchvisionResNetMatrix

MAIN=ROOT/'results_eswa_single';STUDY=MAIN/'extensions/seed-v1'


def expansion_allowed(decision):return decision=='expand_models'


def paired_interval(before,after):
    difference=np.asarray(before,dtype=float)-np.asarray(after,dtype=float)
    rng=np.random.default_rng(20261020)
    draws=difference[rng.integers(0,len(difference),(5000,len(difference)))].mean(axis=1)
    return {'accuracy_drop':float(difference.mean()),'paired_bootstrap_95_interval':np.quantile(draws,[0.025,0.975]).tolist()}


def prepare():
    if STUDY.exists():raise ValueError('Replication already prepared; preserve its state')
    scope=json.loads((MAIN/'extensions/SCOPE_PLAN.json').read_text());tasks=[];deferred=[]
    for block in scope['ablation_task_blocks']:
        cid=block['case_id'];base=MAIN/'audits/formal'/cid/'full/attempt-01';run=base/(cid+'-full')
        paths=sorted(run.glob('models/task/*/training_manifest.json'))
        if not paths:deferred.append({'case_id':cid,'reason':'No retained trained model/accepted construction'});continue
        path=paths[0];manifest=json.loads(path.read_text())
        decision=manifest.get('adaptive_report',{}).get('decision')
        if not expansion_allowed(decision):
            deferred.append({'case_id':cid,'reason':'Original frozen probe did not authorize expansion','original_decision':decision});continue
        request=json.loads((run/'frozen_execution_config.json').read_text());dataset=Path(request['dataset_info']['root'])
        for seed in (43,44):
            tasks.append({'case_id':f's-{len(tasks)+1:02d}','source_case_id':cid,'seed':seed,'source_model_manifest':str(path),
                'source_manifest_sha256':sha(path),'dataset_root':str(dataset),
                'metadata_sha256':sha(dataset/'model_metadata.csv'),'transformation':manifest['screening_transformation'],
                'original_probe_decision':decision,'sample_rule':'ID samples sorted by frozen public ID, exclude ALL original adaptive_probe_sample_ids; same confirmation remainder, not new independent sources.'})
    STUDY.mkdir(parents=True);protocol=STUDY/'protocol';protocol.mkdir();(STUDY/'logs').mkdir()
    write(protocol/'PLAN.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),'tasks':tasks,'deferred':deferred,
        'scope':'Classifier-only replication conditional on original expansion gate, not new Agent audits. No extra profiles to rescue negative/deferred pilots. Original verdicts/intervals unchanged.',
        'seeds':[43,44],'architecture':'resnet18','token_calls':0,'max_wall_seconds_per_profile':4500,
        'statistics':'Within-seed paired image bootstrap5000 seed20261020. Seeds reported separately; no pooling images x seeds as independent audits.',
        'selection':'Predeclared strong border/watermark metadata; original frozen adaptive expansion gate. No selection by Full winning or largest effect.',
        'deadline_utc':json.loads((MAIN/'STUDY_START.json').read_text())['experiment_deadline_utc']})
    identity=source_identity(require_clean=False);archive=archive_source(protocol/'source.zip',identity['source_tree_sha256'])
    write(protocol/'FREEZE.json',{'source_tree_sha256':identity['source_tree_sha256'],'source_archive_sha256':archive,'plan_sha256':sha(protocol/'PLAN.json')})
    print(json.dumps({'status':'FROZEN','new_classifier_profiles':len(tasks),'deferred_blocks':deferred,'api_calls':0}))


def member(index):
    plan=json.loads((STUDY/'protocol/PLAN.json').read_text());task=plan['tasks'][index]
    source=Path(task['source_model_manifest']);dataset=Path(task['dataset_root'])
    if sha(source)!=task['source_manifest_sha256'] or sha(dataset/'model_metadata.csv')!=task['metadata_sha256']:raise ValueError('Frozen member input changed')
    manifest=json.loads(source.read_text());output=STUDY/'audits/formal'/task['case_id']/'full/attempt-01'
    output.mkdir(parents=True,exist_ok=False);run=output/(task['case_id']+'-full');run.mkdir()
    def stage(name):
        write(output/'arm-status.json',{'status':'RUNNING','stage':name})
        with (run/'events.jsonl').open('a') as f:f.write(json.dumps({'stage':name,'time':time.time()})+'\n')
    stage('preparing_fixed_pairs')
    with (dataset/'model_metadata.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    by_id={row['img_id']:row for row in rows};train=[by_id[s] for s in manifest['train_sample_ids']];val=[by_id[s] for s in manifest['validation_sample_ids']]
    probe=set(manifest.get('adaptive_probe_sample_ids',[]))
    test=sorted((row for row in rows if row['split']=='id' and row['img_id'] not in probe),key=lambda row:row['img_id'])
    if not test or {row['img_id'] for row in test}&set(manifest['train_sample_ids']+manifest['validation_sample_ids']):raise ValueError('Invalid fixed confirmation partition')
    transformation=TypeAdapter(ImageInterventionSpec).validate_python(task['transformation'])
    pairs=[]
    for row in test:
        original=dataset/row['img_filename'];changed=run/'pairs'/row['img_id']/'changed.png';changed.parent.mkdir(parents=True)
        quality=InterventionEngine().apply(source_path=original,output_path=changed,transformation=transformation,dataset_root=dataset)
        if not quality.same_dimensions:raise ValueError('Frozen transform changed dimensions')
        noop=changed.parent/'noop.png';Image.fromarray(load_rgb_image_file(original).rgb).save(noop)
        if not np.array_equal(load_rgb_image_file(original).rgb,load_rgb_image_file(noop).rgb):raise ValueError('No-op pixels differ')
        pairs.append({'img_id':row['img_id'],'label':row['y'],'original':str(original),'original_sha256':sha(original),
            'changed':str(changed),'changed_sha256':sha(changed),'noop':str(noop),'quality':quality.model_dump(mode='json')})
    write(run/'PAIR_MANIFEST.json',pairs)
    options={**manifest['training_options'],'architectures':['resnet18'],'seeds':[task['seed']]}
    model=TorchvisionResNetMatrix('fixed-seed-member',options)
    train_paths=[dataset/row['img_filename'] for row in train];val_paths=[dataset/row['img_filename'] for row in val]
    init_path,_=initialization('resnet18')
    key,identity=cache_identity(train_paths,[row['y'] for row in train],val_paths,[row['y'] for row in val],model.options.model_dump(mode='json'),{'resnet18':sha(init_path)})
    cache=MAIN/'shared_classifiers'/key
    if (cache/'TRAINING_IDENTITY.json').exists() and json.loads((cache/'TRAINING_IDENTITY.json').read_text())!=identity:raise ValueError('Cache identity differs')
    if not (cache/'TRAINING_IDENTITY.json').exists():write(cache/'TRAINING_IDENTITY.json',identity)
    stage('training_fixed_seed')
    start=time.monotonic()
    model.fit_matrix(train_paths,[row['y'] for row in train],val_paths,[row['y'] for row in val],run/'models/task/fixed',key,shared_cache=cache)
    train_wall=time.monotonic()-start;logits={};preds={};correct={};labels=np.asarray([row['label'] for row in pairs])
    for name in ('original','changed','noop'):
        stage('inference_'+name);logits[name]=model.predict_logits([Path(row[name]) for row in pairs]);preds[name]=model.classes_[logits[name].argmax(axis=1)];correct[name]=preds[name]==labels
    if not np.array_equal(preds['original'],preds['noop']) or np.max(np.abs(logits['original']-logits['noop']))>1e-5:raise ValueError('No-op inference failed')
    np.savez_compressed(run/'PAIRED_LOGITS.npz',sample_ids=np.asarray([row['img_id'] for row in pairs]),labels=labels,classes=model.classes_,**{name+'_logits':value for name,value in logits.items()})
    write(run/'RESULTS.json',{'seed':task['seed'],'source_case_id':task['source_case_id'],'n_pairs':len(pairs),
        'id_accuracy':float(correct['original'].mean()),'intervention_accuracy':float(correct['changed'].mean()),
        **paired_interval(correct['original'],correct['changed']),'primary_probe_decision_unchanged':task['original_probe_decision'],
        'noop_max_logit_difference':float(np.max(np.abs(logits['original']-logits['noop']))),
        'training_and_validation_wall_seconds':train_wall,'gpu_kernel_active_seconds':None,
        'profiles':[{k:v for k,v in profile.items() if k!='checkpoint_path'} for profile in model.profiles],
        'scope':plan['scope'],'api_calls':0,'new_agent_audits':0})
    write(output/'arm-status.json',{'status':'COMPLETED','stage':'completed'})


def run():
    import msvcrt
    lock=(MAIN/'extension.queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Another GPU extension is active')
    protocol=STUDY/'protocol';plan=json.loads((protocol/'PLAN.json').read_text());frozen=json.loads((protocol/'FREEZE.json').read_text())
    if source_identity(require_clean=False)['source_tree_sha256']!=frozen['source_tree_sha256'] or sha(protocol/'PLAN.json')!=frozen['plan_sha256'] or sha(protocol/'source.zip')!=frozen['source_archive_sha256']:raise ValueError('Frozen replication source/plan changed')
    status=STUDY/'TASK_STATUS.csv'
    if status.exists():raise ValueError('Replication already scheduled; inspect before resuming')
    rows=[{'case_id':task['case_id'],'variant':'full','phase':'formal','status':'PENDING','attempt':1,'study_root':str(STUDY),
        'seed':task['seed'],'pid':'','reason':'','started_utc':'','finished_utc':''} for task in plan['tasks']]
    def persist(stage):
        if rows:
            with status.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(MAIN/'LIVE_STATUS.json',{'stage':stage,'queue_pid':os.getpid(),'formal_started':True,'tasks':rows,
            'updated_utc':datetime.now(timezone.utc).isoformat(),'next_gate':'No model/API audit retries. Complete fixed-authorized seeds, then final evidence consolidation and result package.'})
    deadline=datetime.fromisoformat(plan['deadline_utc'])
    for index,row in enumerate(rows):
        if datetime.now(timezone.utc)+timedelta(seconds=4500)>deadline-timedelta(hours=12) or datetime.now(timezone.utc)>deadline-timedelta(hours=24):
            row['reason']='Deadline whole-profile guard';persist('seed_scope_stop_requires_export');return
        with (STUDY/'logs'/(row['case_id']+'.stdout.log')).open('xb') as out,(STUDY/'logs'/(row['case_id']+'.stderr.log')).open('xb') as err:
            process=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__)),'member','--index',str(index)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            row.update(status='RUNNING',pid=process.pid,started_utc=datetime.now(timezone.utc).isoformat());persist('seed_replication_running')
            try:code=process.wait(timeout=4500)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);process.wait();code=124
        row.update(status='COMPLETED' if code==0 else 'FAILED',finished_utc=datetime.now(timezone.utc).isoformat(),reason='' if code==0 else 'Member failed; retain artifacts')
        if code!=0:persist('seed_replication_requires_repair');return
    persist('seed_replication_complete_requires_final_export')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','member']);parser.add_argument('--index',type=int);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run()
    else:member(args.index)
