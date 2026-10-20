"""Freeze a previously inspected real-data case and verify exact historical checkpoint reuse."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.eswa_20261019.run_one import cache_identity
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.waterbirds_paper import sha,initialization,source_identity,archive_source

MAIN=ROOT/'results_eswa_single';STUDY=MAIN/'extensions/pet-v1'
OLD=ROOT/'runs/e3-representative-20261014'


def validate_recipe(old,new):
    left={key:value for key,value in old.items() if key not in {'architectures','seeds'}}
    right={key:value for key,value in new.items() if key not in {'architectures','seeds'}}
    if left!=right:raise ValueError('Historical per-profile training recipe differs')
    if not set(new['architectures']).issubset(old['architectures']) or not set(new['seeds']).issubset(old['seeds']):
        raise ValueError('Historical matrix does not contain requested profile')


def main():
    if STUDY.exists():raise ValueError('Pet study already prepared; inspect before resuming')
    with (MAIN/'extensions/fixed-candidate-v1/TASK_STATUS.csv').open(newline='',encoding='utf-8-sig') as f:arms=list(csv.DictReader(f))
    if len(arms)!=6 or any(arm['status'] not in {'COMPLETED','FAILED_RETAINED'} for arm in arms):raise ValueError('Controlled collection still active')
    profile=json.loads((MAIN/'protocol/formal-main-v1/profile.json').read_text())
    options=profile['execution_options']['training_options']
    acceptance=json.loads((OLD/'intake-acceptance.json').read_text())
    if sha(ROOT/'shortcutminer/task_models/torchvision_matrix.py')!=acceptance['scientific_source_hashes']['shortcutminer/task_models/torchvision_matrix.py']:
        raise ValueError('Historical training implementation changed; refuse implicit reuse')
    model=OLD/'case-e3-representative-v1/models/task/TM-5607d27ac118c0c7c217e75d'
    manifest=json.loads((model/'training_manifest.json').read_text())
    bundle=json.loads((model/'profiles.json').read_text())
    validate_recipe(manifest['training_options'],options)
    selected=next(row for row in bundle['profiles'] if row['architecture']=='resnet18' and row['seed']==42)
    weights=OLD/'case-e3-representative-v1'/selected['weights_reference']
    if sha(weights)!=selected['weights_sha256']:raise ValueError('Historical checkpoint hash mismatch')
    discovery=OLD/'public-discovery';training=OLD/'public-training';old_input=OLD/'case-e3-representative-v1-input'
    def rows(root,filename):
        with (root/filename).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    public=rows(training,'metadata.csv');old_rows=rows(old_input,'model_metadata.csv');observations=rows(discovery,'metadata.csv')
    if sorted({row['y'] for row in public})!=bundle['classes']:raise ValueError('Historical class mapping differs')
    by_id={row['img_id']:row for row in public};historical={row['img_id']:row for row in old_rows}
    train=[row for row in public if row['split']=='train'];val=[row for row in public if row['split']=='val']
    fitting_ids=set(manifest['train_sample_ids'])|set(manifest['validation_sample_ids'])
    if [row['img_id'] for row in train]!=manifest['train_sample_ids'] or [row['img_id'] for row in val]!=manifest['validation_sample_ids']:
        raise ValueError('Historical training/validation order differs')
    if {row['img_id'] for row in observations}!={row['img_id'] for row in train} or any(row['split']!='train' for row in observations):
        raise ValueError('Discovery must contain only exact training partition')
    hashes={str(training/'metadata.csv'):sha(training/'metadata.csv'),str(discovery/'metadata.csv'):sha(discovery/'metadata.csv')}
    for row in public:
        path=training/row['img_filename'];digest=sha(path);hashes[str(path)]=digest
        if row['img_id'] in fitting_ids:
            old=historical[row['img_id']]
            if old['y']!=row['y'] or sha(old_input/old['img_filename'])!=digest:raise ValueError('Historical training bytes/labels differ')
    for row in observations:
        path=discovery/row['img_filename'];digest=sha(path);hashes[str(path)]=digest
        original=by_id[row['img_id']]
        if row['y']!=original['y'] or digest!=hashes[str(training/original['img_filename'])]:raise ValueError('Discovery bytes/labels differ')
    init_path,init=initialization('resnet18')
    if sha(init_path)!=selected['initialization']['sha256']:raise ValueError('Historical initialization changed')
    key,content=cache_identity([training/row['img_filename'] for row in train],[row['y'] for row in train],
        [training/row['img_filename'] for row in val],[row['y'] for row in val],options,{'resnet18':sha(init_path)})
    cache=MAIN/'shared_classifiers'/key;record=cache/'TRAINING_IDENTITY.json'
    if cache.exists():raise ValueError('Pet import cache already exists; inspect provenance')
    cache.mkdir(parents=True);write(record,content)
    member=cache/'resnet18-seed42';member.mkdir();shutil.copyfile(weights,member/'best.npz')
    write(member/'profile.json',{**{key:value for key,value in selected.items() if key!='weights_reference'},'classes':bundle['classes']})
    write(cache/'HISTORICAL_IMPORT.json',{'source_weights':str(weights),'weights_sha256':selected['weights_sha256'],
        'profile_source':str(model/'profiles.json'),'profile_source_sha256':sha(model/'profiles.json'),
        'training_manifest_sha256':sha(model/'training_manifest.json'),'training_source_matches':True,
        'training_validation_bytes_labels_order_match':True,'same_per_profile_recipe':True,
        'scope':'Existing historical model, not a newly trained independent classifier. Heldout data previously inspected.'})
    STUDY.mkdir(parents=True);protocol=STUDY/'protocol';protocol.mkdir();(STUDY/'logs').mkdir()
    write(protocol/'profile.json',profile)
    write(STUDY/'FORMAL_PUBLIC_INPUTS.json',{'cases':[{'case_id':'r-01','development':False,
        'discovery':str(discovery),'training':str(training),'metadata_sha256':sha(training/'metadata.csv'),
        'task':'37-class image classification. Target y is an opaque categorical class ID. Discover candidate dataset shortcuts from approved training evidence.'}]})
    write(protocol/'input-hashes.json',hashes)
    write(protocol/'RULES.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),
        'case':'Pet real-data reanalysis','previously_inspected_data':True,'new_independent_confirmation_pool':False,
        'historical_checkpoint_import':str(cache/'HISTORICAL_IMPORT.json'),'shared_cache_key':key,
        'token_stop':600000,'wall_seconds':4500,'global_planning_envelope':16000000,
        'architecture':'resnet18','seed':42,'training_recipe':options,
        'scope':'One Full audit, preserved five-candidate dispositions where returned; old low-pass finding remains texture sensitivity, not causal nuisance attribution.',
        'local_diagnostics_plan':{'sample_rule':'At most500 historical ID images, seed20261020 fixed independently of outcomes.',
            'background':'Official trimaps as evaluator-provided masks, foreground/boundary unchanged; background blur plus matched-area foreground control and exact no-op. Oracle-assisted local diagnostic, not autonomous segmentation success.',
            'text':'Perform only if a reproducible localized text mask and fidelity check are available; otherwise defer with actual reason. Never equate global filtering with text removal.',
            'scope':'Same historical checkpoint/pool; exploratory paired sensitivity, no new blind confirmation or human semantic certification.'},
        'human_adjudication':'PENDING','source_policy':'Separate Pet version; main and controlled extension archives/results immutable.'})
    identity=source_identity(require_clean=False);archive=archive_source(protocol/'source.zip',identity['source_tree_sha256'])
    write(protocol/'FREEZE.json',{'source_tree_sha256':identity['source_tree_sha256'],'source_archive_sha256':archive,
        'files':{name:sha(protocol/name) for name in ('profile.json','input-hashes.json','RULES.json','source.zip')},
        'public_index_sha256':sha(STUDY/'FORMAL_PUBLIC_INPUTS.json')})
    print(json.dumps({'status':'PET_FROZEN','training_samples':len(train),'validation_samples':len(val),
        'historical_id_samples':sum(row['split']=='id' for row in public),'historical_checkpoint_imported':True,
        'new_training_runs':0,'independent_confirmation':False}))


if __name__=='__main__':main()
