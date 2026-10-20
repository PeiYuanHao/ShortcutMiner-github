"""Posthoc evaluator-only specificity audit and exact implanted-cue removal diagnostics."""
import argparse,csv,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.e4_ablation.variants.policies import write
from experiments.eswa_20261019.prepare_inputs import sources,transform
from experiments.eswa_20261019.run_one import cache_identity
from experiments.eswa_20261019.pet_local_ops import central_mask,paired_effects
from shortcutminer.experiment.waterbirds_paper import sha,initialization,source_identity,archive_source
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.task_models.torchvision_matrix import TorchvisionResNetMatrix

MAIN=ROOT/'results_eswa_single';STUDY=MAIN/'extensions/synthetic-specificity-v1'


def source_regions(record,private):
    _,support=transform(np.zeros((32,32,3),dtype=np.uint8),private['spec']['mechanism'],record['state'],private['geometry'])
    content=np.zeros((40,40),dtype=bool);content[4:36,4:36]=True
    return support,content


def exact_pairs(original,pristine,record,private):
    support,content=source_regions(record,private)
    base=np.full((40,40,3),127,dtype=np.uint8);base[4:36,4:36]=pristine
    expected,_=transform(pristine,private['spec']['mechanism'],record['state'],private['geometry'])
    if not np.array_equal(original,expected):raise ValueError('Private source reconstruction mismatch')
    changed=original.copy();changed[support]=base[support]
    if not np.array_equal(changed,base) or not np.array_equal(changed[~support],original[~support]):raise ValueError('Exact oracle removal violated source identity')
    eligible=content&~support;count=int(min(support.sum(),eligible.sum()))
    control_mask=central_mask(eligible,count);blur=np.asarray(Image.fromarray(original).filter(ImageFilter.GaussianBlur(3)))
    control=np.where(control_mask[...,None],blur,original)
    return {'original':original,'noop':original.copy(),'removed':changed,'content_control':control}, {
        'cue_support_pixels':int(support.sum()),'oracle_changed_pixels':int(np.any(changed!=original,axis=2).sum()),
        'oracle_noncue_pixels_changed':0,'control_planned_pixels':count,
        'control_changed_pixels':int(np.any(control!=original,axis=2).sum()),
        'control_scope':'Matched intended mask count in non-cue source-content region; a damaging control, not semantic preservation.'}


def audit_actual(output):
    records=[]
    controlled=json.loads((MAIN/'extensions/fixed-candidate-v1/protocol/CONTROLLED_INPUTS.json').read_text())
    aliases={row['case_id']:row['source_case_id'] for row in controlled['arms']}
    for study,root,status in [('main',MAIN,MAIN/'FORMAL_TASK_STATUS.csv'),
        ('controlled',MAIN/'extensions/fixed-candidate-v1',MAIN/'extensions/fixed-candidate-v1/TASK_STATUS.csv')]:
        with status.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        for row in rows:
            cid=aliases.get(row['case_id'],row['case_id']);private=json.loads((MAIN/'private/formal'/(cid+'.json')).read_text())
            by_id={item['img_id']:item for item in private['records']}
            run=root/'audits/formal'/row['case_id']/'full/attempt-01'/(row['case_id']+'-full') if study=='controlled' else root/'audits/formal'/row['case_id']/row['variant']/'attempt-01'/(row['case_id']+'-'+row['variant'].replace('_','-'))
            config_path=run/'frozen_execution_config.json'
            if not config_path.exists():continue
            dataset=Path(json.loads(config_path.read_text())['dataset_info']['root'])
            for path in run.glob('interventions/**/intervention_manifest.json'):
                manifest=json.loads(path.read_text());changed_total=outside_total=content_total=0;valid=0
                for pair in manifest.get('pairs',[]):
                    source=dataset/pair['original_reference'];target=run/pair['counterfactual_reference']
                    if not source.is_file() or not target.is_file():continue
                    original=load_rgb_image_file(source).rgb;altered=load_rgb_image_file(target).rgb
                    if original.shape!=altered.shape:raise ValueError('Retained intervention shape mismatch')
                    item=by_id[pair['original_sample_id']]
                    if sha(source)!=item['public_file_sha256']:raise ValueError('Retained original source identity differs')
                    support,content=source_regions(item,private);changed=np.any(original!=altered,axis=2)
                    changed_total+=int(changed.sum());outside_total+=int((changed&~support).sum());content_total+=int((changed&content&~support).sum());valid+=1
                records.append({'study':study,'case_id':row['case_id'],'source_case_id':cid,'condition':row.get('condition') or row['variant'],
                    'candidate_id':manifest['candidate_id'],'pairs_checked':valid,'changed_pixels_total':changed_total,
                    'noncue_pixels_changed_total':outside_total,'noncue_source_content_pixels_changed_total':content_total,
                    'exact_noncue_source_content_preserved':content_total==0 if valid else None,'manifest_path':str(path),'manifest_sha256':sha(path),
                    'limitation':'Pixel support specificity, not human semantic judgment; deviations are retained, never repaired retroactively.'})
    write(output/'AUTONOMOUS_PIXEL_SPECIFICITY.json',records)
    with (output/'AUTONOMOUS_PIXEL_SPECIFICITY.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return records


def main():
    if STUDY.exists():raise ValueError('Specificity diagnostics already scheduled; preserve prior run')
    STUDY.mkdir(parents=True);protocol=STUDY/'protocol';protocol.mkdir()
    attempt=STUDY/'audits/formal/oracle-01/full/attempt-01';run=attempt/'oracle-01-full';run.mkdir(parents=True)
    row={'case_id':'oracle-01','variant':'full','phase':'formal','status':'RUNNING','attempt':1,'study_root':str(STUDY),'pid':os.getpid()}
    def persist(stage):
        write(attempt/'arm-status.json',{'status':row['status'],'stage':stage})
        with (run/'events.jsonl').open('a') as f:f.write(json.dumps({'stage':stage,'time':time.time()})+'\n')
        with (STUDY/'TASK_STATUS.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        write(MAIN/'LIVE_STATUS.json',{'stage':'synthetic_specificity_running' if row['status']=='RUNNING' else
            ('synthetic_specificity_complete_requires_final_export' if row['status']=='COMPLETED' else 'synthetic_specificity_requires_repair'),
            'queue_pid':os.getpid(),'formal_started':True,'tasks':[row],'updated_utc':datetime.now(timezone.utc).isoformat(),
            'next_gate':'Retain autonomous specificity failures. Exact implanted-cue removal is evaluator-only, not Agent success. Consolidate and package without positive-result retries.'})
    start=time.monotonic();persist('freezing_all_condition_diagnostic')
    try:
        public=json.loads((MAIN/'FORMAL_PUBLIC_INPUTS.json').read_text())['cases']
        plan={'recorded_utc':datetime.now(timezone.utc).isoformat(),'scope':'Evaluator-only posthoc measurement-validity checks, never substituted for autonomous intervention results or a new blind discovery.',
            'cases':[case['case_id'] for case in public],'sample_rule':'Same500 pristine source SHA IDs from sorted ID pool, permutation seed20261020. Previously used source pool; may include former probes, not an untouched confirmation set.',
            'checkpoint_rule':'Only existing resnet18/42 cache with exact bytes/labels/recipe/init identity. Missing cache => deferred; no new training.',
            'operations':'Exact private implanted-support restoration to pristine original RGB+neutral127 padding; no-op; matched-count non-cue source-content Gaussian3 control. All conditions included, independent of effects/winners.',
            'fidelity':'Exact removal must equal pristine canvas and leave all noncue pixels untouched; controls damage source content intentionally. Existing broad interventions audited but not rewritten.',
            'statistics':'Within-condition paired5000 bootstrap; exploratory pointwise95 and within-condition Bonferroni3 contrasts, not familywise confirmation over all cases or independent-dataset inference.',
            'api_calls':0,'new_training_runs':0,'human_adjudication':'PENDING'}
        write(protocol/'PLAN.json',plan)
        identity=source_identity(require_clean=False);archive=archive_source(protocol/'source.zip',identity['source_tree_sha256'])
        write(protocol/'FREEZE.json',{'source_tree_sha256':identity['source_tree_sha256'],'source_archive_sha256':archive,'plan_sha256':sha(protocol/'PLAN.json')})
        persist('auditing_retained_autonomous_pixels');audit=audit_actual(run)
        original_rows,source_manifest=sources(Path('E:/datasets/E1/cifar-10-python/cifar-10-batches-py'))
        pristine={item['sha256']:item['pixels'] for item in original_rows};write(protocol/'OFFICIAL_SOURCES.json',source_manifest)
        first=json.loads((MAIN/'private/formal'/(public[0]['case_id']+'.json')).read_text())
        source_ids=sorted(item['source_sha256'] for item in first['records'] if item['split']=='id')
        rng=np.random.default_rng(20261020);selected=[source_ids[int(index)] for index in rng.permutation(len(source_ids))[:500]]
        write(protocol/'SELECTED_SOURCE_IDS.json',selected)
        options=json.loads((MAIN/'protocol/formal-main-v1/profile.json').read_text())['execution_options']['training_options']
        init_path,_=initialization('resnet18');results=[];deferred=[]
        for case in public:
            cid=case['case_id'];private=json.loads((MAIN/'private/formal'/(cid+'.json')).read_text());by_source={item['source_sha256']:item for item in private['records']}
            dataset=Path(case['training'])
            with (dataset/'metadata.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            by_id={item['img_id']:item for item in rows}
            train=sorted((item for item in rows if item['split']=='train'),key=lambda item:item['img_id']);val=sorted((item for item in rows if item['split']=='val'),key=lambda item:item['img_id'])
            key,_=cache_identity([dataset/item['img_filename'] for item in train],[item['y'] for item in train],
                [dataset/item['img_filename'] for item in val],[item['y'] for item in val],options,{'resnet18':sha(init_path)})
            cache=MAIN/'shared_classifiers'/key
            if not (cache/'resnet18-seed42/profile.json').exists():deferred.append({'case_id':cid,'reason':'No exact existing resnet18/42 cache; no training permitted'});continue
            member=json.loads((cache/'resnet18-seed42/profile.json').read_text());weights=cache/'resnet18-seed42/best.npz'
            if sha(weights)!=member['weights_sha256']:raise ValueError('Cached checkpoint identity differs')
            case_dir=run/cid;case_dir.mkdir();pairs=[]
            for index,source_id in enumerate(selected):
                record=by_source[source_id];item=by_id[record['img_id']];source=dataset/item['img_filename']
                if sha(source)!=record['public_file_sha256']:raise ValueError('Public image changed')
                images,fidelity=exact_pairs(load_rgb_image_file(source).rgb,pristine[source_id],record,private)
                pair={'img_id':item['img_id'],'source_sha256':source_id,'y':item['y'],'fidelity':fidelity}
                for name,pixels in images.items():
                    target=case_dir/'images'/item['img_id']/(name+'.png');target.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pixels).save(target)
                    pair[name]=str(target);pair[name+'_sha256']=sha(target)
                pairs.append(pair)
                if (index+1)%100==0:persist(cid+'_pairs_'+str(index+1))
            write(case_dir/'PAIR_MANIFEST.json',pairs)
            model=TorchvisionResNetMatrix('evaluator-specificity',options);model.classes_=np.asarray(member['classes'])
            model.profiles=[{**member,'checkpoint_path':weights}];logits={};preds={};correct={};labels=np.asarray([item['y'] for item in pairs])
            for name in ('original','noop','removed','content_control'):
                persist(cid+'_inference_'+name);logits[name]=model.predict_logits([Path(item[name]) for item in pairs]);preds[name]=model.classes_[logits[name].argmax(axis=1)];correct[name]=preds[name]==labels
            noop_delta=float(np.max(np.abs(logits['original']-logits['noop'])))
            if noop_delta>1e-5 or not np.array_equal(preds['original'],preds['noop']):raise ValueError('No-op prediction invariance failed')
            effects=paired_effects({'original':correct['original'],'background':correct['removed'],'foreground_control':correct['content_control']})
            result={'case_id':cid,'private_mechanism':private['spec']['mechanism'],'rho':private['spec']['label_cue_match_rho'],
                'n_pairs':len(pairs),'checkpoint_sha256':member['weights_sha256'],'cache_identity':key,
                'accuracy':{name:float(value.mean()) for name,value in correct.items()},
                'effects':{'oracle_cue_removal_drop':effects['background_accuracy_drop'],'source_content_control_drop':effects['foreground_control_accuracy_drop'],
                    'oracle_minus_content_drop':effects['background_minus_foreground_drop']},'noop_max_logit_difference':noop_delta,
                'scope':plan['scope'],'human_adjudication':'PENDING'}
            write(case_dir/'RESULTS.json',result);results.append(result)
            np.savez_compressed(case_dir/'PAIRED_LOGITS.npz',sample_ids=np.asarray([item['img_id'] for item in pairs]),labels=labels,classes=model.classes_,**{name+'_logits':value for name,value in logits.items()})
        write(run/'RESULTS.json',{'cases':results,'deferred':deferred,'audited_autonomous_interventions':len(audit),
            'autonomous_interventions_changing_noncue_source_content':sum(item['exact_noncue_source_content_preserved'] is False for item in audit),
            'api_calls':0,'new_training_runs':0,'scope':plan['scope']})
        write(attempt/'OFFLINE_RECEIPT.json',{'status':'COMPLETED','wall_seconds':time.monotonic()-start,'api_calls':0,'new_training_runs':0,
            'cases_completed':len(results),'cases_deferred':len(deferred),'autonomous_results_unchanged':True})
        row['status']='COMPLETED';persist('completed')
    except Exception as error:
        write(attempt/'OFFLINE_FAILURE.json',{'type':type(error).__name__,'message':str(error),'api_calls':0,'new_training_runs':0})
        row['status']='FAILED';persist('failed');raise


if __name__=='__main__':main()
