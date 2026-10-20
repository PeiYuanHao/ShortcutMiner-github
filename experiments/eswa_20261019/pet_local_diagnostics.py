"""Zero-API historical-checkpoint local diagnostics. No training, oracle success relabeling or audit retry."""
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.e4_ablation.variants.policies import write
from experiments.eswa_20261019.pet_local_ops import local_pairs,paired_effects
from shortcutminer.experiment.waterbirds_paper import sha,source_identity,archive_source
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.task_models.torchvision_matrix import TorchvisionResNetMatrix

MAIN=ROOT/'results_eswa_single';STUDY=MAIN/'extensions/pet-local-v1'
PET=MAIN/'extensions/pet-v1';OLD=ROOT/'runs/e3-representative-20261014'
ORIGINAL=Path('E:/datasets/E3/images');TRIMAPS=Path('E:/datasets/E3/annotations/annotations/trimaps')


def csv_write(path,rows,fields):
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    if STUDY.exists():raise ValueError('Local diagnostics already scheduled; preserve prior attempt')
    with (PET/'TASK_STATUS.csv').open(newline='',encoding='utf-8-sig') as f:pet=list(csv.DictReader(f))
    if len(pet)!=1 or pet[0]['status'] not in {'COMPLETED','FAILED_RETAINED'}:raise ValueError('Pet Agent worker must finish first')
    STUDY.mkdir(parents=True);protocol=STUDY/'protocol';protocol.mkdir()
    output=STUDY/'audits/formal/local-01/full/attempt-01';output.mkdir(parents=True)
    run=output/'local-01-full';run.mkdir()
    row={'case_id':'local-01','variant':'full','phase':'formal','status':'RUNNING','attempt':1,
        'study_root':str(STUDY),'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'finished_utc':'','reason':'Evaluator-only local sensitivity, not an Agent audit'}
    def persist(stage):
        write(output/'arm-status.json',{'status':row['status'],'stage':stage,'framework':'zero-API offline diagnostic'})
        with (run/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'timestamp':datetime.now(timezone.utc).isoformat(),'stage':stage})+'\n')
        with (STUDY/'TASK_STATUS.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        write(MAIN/'LIVE_STATUS.json',{'stage':'pet_local_diagnostics_running' if row['status']=='RUNNING' else
            ('pet_local_diagnostics_complete_requires_analysis_and_export' if row['status']=='COMPLETED' else 'pet_local_diagnostics_requires_repair'),
            'queue_pid':os.getpid(),'formal_started':True,'tasks':[row],'updated_utc':datetime.now(timezone.utc).isoformat(),
            'next_gate':'Preserve audit failure and standalone evaluator diagnostics separately; next analyze retained paired records/coverage/cost and export.'})
    begin=time.monotonic();persist('freezing_local_plan')
    try:
        with (OLD/'public-training/metadata.csv').open(newline='',encoding='utf-8-sig') as f:all_rows=list(csv.DictReader(f))
        pool=sorted((item for item in all_rows if item['split']=='id'),key=lambda item:item['img_id'])
        rng=np.random.default_rng(20261020);selected=[pool[int(index)] for index in rng.permutation(len(pool))[:500]]
        pet_rules=json.loads((PET/'protocol/RULES.json').read_text())
        cache=MAIN/'shared_classifiers'/pet_rules['shared_cache_key']
        profile=json.loads((cache/'resnet18-seed42/profile.json').read_text());weights=cache/'resnet18-seed42/best.npz'
        if sha(weights)!=profile['weights_sha256']:raise ValueError('Frozen checkpoint changed')
        plan={'recorded_utc':datetime.now(timezone.utc).isoformat(),'pet_rules_sha256':sha(PET/'protocol/RULES.json'),
            'sample_selection':'Sorted historical ID IDs, permutation seed20261020, first500; no replacement after missingness or quality deferral.',
            'selected_rows':selected,'image_count_planned':len(selected),'checkpoint_sha256':profile['weights_sha256'],
            'new_training_runs':0,'api_calls':0,'mask_source':'Official Oxford-IIIT trimaps:1 foreground,2 background,3 unclassified. Evaluator-only annotations, never Agent inputs.',
            'operator':'RGB Gaussian blur radius3 native pixels; erode both eligible masks3 pixels; choose deepest min(FG,BG) pixels in each with stable flat-index ties; require64 safe pixels.',
            'fidelity':'Background edit leaves exact foreground/boundary pixels unchanged. Foreground control leaves background/boundary unchanged. Match planned eligible pixel count; report actual changed count separately.',
            'noop':'Decode original pixels and write identical lossless RGB PNG; require exact pixels, same class predictions and max logit difference<=1e-5.',
            'statistics':'Paired image-level bootstrap5000 seed20261020, signed accuracy drops; three exploratory contrasts, pointwise95% and Bonferroni three-contrast intervals. No optional stopping or outcome-driven selection.',
            'limits':'Previously inspected pool/checkpoint, evaluator-provided segmentation, possible same-object correlations not identified. Pixel invariants do not certify semantic preservation. No causal nuisance or new blind discovery claim.',
            'text_localization':'DEFERRED: no validated localized text masks for these opaque historical records; global blur is not text removal.',
            'human_adjudication':'PENDING'}
        write(protocol/'PLAN.json',plan)
        identity=source_identity(require_clean=False)
        archive=archive_source(protocol/'source.zip',identity['source_tree_sha256'])
        write(protocol/'FREEZE.json',{'source_tree_sha256':identity['source_tree_sha256'],'source_archive_sha256':archive,
            'plan_sha256':sha(protocol/'PLAN.json'),'checkpoint_sha256':sha(weights)})
        persist('indexing_official_image_hashes')
        by_hash={}
        for path in sorted(ORIGINAL.glob('*.jpg')):
            if path.name.startswith('._'):continue
            by_hash.setdefault(sha(path),[]).append(path)
        input_hashes={};valid=[];deferred=[];seen=set()
        root=OLD/'public-training'
        for index,item in enumerate(selected):
            path=root/item['img_filename'];digest=sha(path);input_hashes[str(path)]=digest
            matches=by_hash.get(digest,[])
            masks=[TRIMAPS/(source.stem+'.png') for source in matches]
            if not masks or any(not mask.is_file() for mask in masks):
                deferred.append({'img_id':item['img_id'],'reason':'Missing exact-byte original/trimap'});continue
            mask_hashes={sha(mask) for mask in masks}
            if len(mask_hashes)!=1 or digest in seen:
                deferred.append({'img_id':item['img_id'],'reason':'Ambiguous masks or duplicate source bytes'});continue
            seen.add(digest);mask=masks[0];input_hashes[str(mask)]=sha(mask)
            rgb=load_rgb_image_file(path).rgb
            with Image.open(mask) as image:trimap=np.asarray(image).copy()
            try:images,fidelity=local_pairs(rgb,trimap)
            except ValueError as error:
                deferred.append({'img_id':item['img_id'],'reason':str(error)});continue
            record={**item,'original_path':str(path),'original_sha256':digest,'trimap_path':str(mask),
                'trimap_sha256':sha(mask),'fidelity':fidelity}
            for name,pixels in images.items():
                target=run/'images'/item['img_id']/(name+'.png');target.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(pixels).save(target)
                if not np.array_equal(load_rgb_image_file(target).rgb,pixels):raise ValueError('Lossless output pixel identity changed')
                record[name+'_path']=str(target);record[name+'_sha256']=sha(target)
            valid.append(record)
            if (index+1)%50==0:persist('preparing_pairs_'+str(index+1))
        write(run/'PAIR_MANIFEST.json',{'selected':len(selected),'valid':valid,'deferred':deferred})
        write(protocol/'input-hashes.json',input_hashes)
        if not valid:raise ValueError('No usable annotated pairs; no invented results')
        model=TorchvisionResNetMatrix('pet-local-historical',pet_rules['training_recipe'])
        model.classes_=np.asarray(profile['classes'],dtype=str)
        model.profiles=[{**profile,'checkpoint_path':weights,'shared_current_run_reuse':True}]
        logits={};predictions={};correct={};labels=np.asarray([item['y'] for item in valid],dtype=str)
        for name in ('original','noop','background','foreground_control'):
            persist('inference_'+name)
            logits[name]=model.predict_logits([Path(item[name+'_path']) for item in valid])
            predictions[name]=model.classes_[logits[name].argmax(axis=1)]
            correct[name]=predictions[name]==labels
        max_delta=float(np.max(np.abs(logits['original']-logits['noop'])))
        if max_delta>1e-5 or not np.array_equal(predictions['original'],predictions['noop']):raise ValueError('No-op model invariance failed; effects not accepted')
        for path,digest in input_hashes.items():
            if sha(Path(path))!=digest:raise ValueError('Diagnostic source changed during computation')
        persist('paired_bootstrap')
        effects=paired_effects(correct)
        np.savez_compressed(run/'PAIRED_LOGITS.npz',sample_ids=np.asarray([item['img_id'] for item in valid]),labels=labels,
            **{name+'_logits':array for name,array in logits.items()})
        prediction_rows=[{'img_id':item['img_id'],'label':item['y'],
            **{name+'_prediction':str(predictions[name][index]) for name in predictions},
            **{name+'_correct':bool(correct[name][index]) for name in correct}} for index,item in enumerate(valid)]
        csv_write(run/'PAIRED_PREDICTIONS.csv',prediction_rows,list(prediction_rows[0]))
        write(run/'RESULTS.json',{'scope':plan['limits'],'n_selected':len(selected),'n_valid':len(valid),'n_deferred':len(deferred),
            'accuracy':{name:float(values.mean()) for name,values in correct.items()},'paired_effects':effects,
            'noop_max_abs_logit_delta':max_delta,'fidelity':'Exact pixel invariants verified for every valid pair; not human semantic certification.',
            'text_localization':plan['text_localization'],'human_adjudication':'PENDING','api_calls':0,'new_training_runs':0})
        write(output/'OFFLINE_RECEIPT.json',{'status':'COMPLETED','wall_seconds':time.monotonic()-begin,
            'api_calls':0,'new_training_runs':0,'checkpoint_sha256':profile['weights_sha256'],
            'n_valid':len(valid),'scope':'Posthoc evaluator-assisted diagnostic, separate from failed Pet Agent audit.'})
        row.update(status='COMPLETED',finished_utc=datetime.now(timezone.utc).isoformat());persist('completed')
    except Exception as error:
        write(output/'OFFLINE_FAILURE.json',{'error_type':type(error).__name__,'message':str(error),
            'wall_seconds':time.monotonic()-begin,'api_calls':0,'new_training_runs':0})
        row.update(status='FAILED',finished_utc=datetime.now(timezone.utc).isoformat(),reason=str(error));persist('failed')
        raise


if __name__=='__main__':main()
