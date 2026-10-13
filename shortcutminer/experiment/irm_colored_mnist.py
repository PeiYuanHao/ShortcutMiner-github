"""Read-only IRM RGB export adapter; training-only discovery and content isolation."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from shortcutminer.experiment.colorized_mnist import partition
from shortcutminer.experiment.waterbirds_paper import sha, identity, write

def inspect_source(source):
    source = Path(source).resolve(strict=True)
    records = []
    with (source/'metadata.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            path = (source/row['img_filename']).resolve(strict=True)
            path.relative_to(source)
            if row['y'] not in {'0','1'} or row['color'] not in {'red','green'}:
                raise RuntimeError('Expected binary labels and red/green colors')
            expected = 'train' if row['environment'] in {'train_env1','train_env2'} else 'test' if row['environment']=='ood_env' else None
            if expected is None or row['split'] != expected:
                raise RuntimeError('Unknown IRM environment or inconsistent split')
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (14,14):
                    raise RuntimeError('Expected canonical IRM RGB 14x14 export')
                rgb = np.array(image)
            channel = 0 if row['color']=='red' else 1
            if rgb[:,:,2].any() or rgb[:,:,1-channel].any() or not rgb[:,:,channel].any():
                raise RuntimeError('Color metadata disagrees with foreground pixels')
            records.append({'reference':row['img_filename'],'sample_id':row['img_id'],'label':row['y'],
                'color':row['color'],'environment':row['environment'],'original_split':'training' if expected=='train' else 'testing',
                'file_sha256':sha(path),'digit_intensity_sha256':hashlib.sha256(rgb.max(2).tobytes()).hexdigest()})
    if len(records)!=60000 or len({r['sample_id'] for r in records})!=60000:
        raise RuntimeError('Expected 60000 unique IRM samples')
    return records

def prepare(config):
    source = Path(config['dataset_root']).resolve(strict=True)
    records = inspect_source(source)
    source_hash = identity(records)
    if source_hash != config['expected_source_inventory_sha256']:
        raise RuntimeError('IRM source changed from frozen inventory')
    assignments, excluded = partition(records, config['split_protocol'])
    root = Path(config['derived_dataset_root']).resolve()
    fingerprint = identity({'source_inventory_sha256':source_hash,'split_protocol':config['split_protocol'],'adapter':'irm_rgb_v1'})
    manifest_path = root/'input_manifest.json'
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing['input_hash'] != fingerprint: raise RuntimeError('Frozen IRM input protocol changed')
        for reference, expected in existing['view_hashes'].items():
            if sha(root/reference)!=expected: raise RuntimeError('Frozen IRM metadata changed')
        for r in records:
            if assignments.get(r['reference']) and sha(root/r['reference'])!=r['file_sha256']:
                raise RuntimeError('Frozen IRM image changed')
        return root, existing
    root.mkdir(parents=True,exist_ok=True)
    rows, mapping = [], []
    for r in records:
        split = assignments.get(r['reference'],'excluded_overlap')
        mapping.append({**r,'execution_split':split})
        if split == 'excluded_overlap': continue
        target = root/r['reference']; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): os.link(source/r['reference'],target)
        if sha(target)!=r['file_sha256']: raise RuntimeError('IRM derived image differs')
        rows.append({'img_id':r['sample_id'],'img_filename':r['reference'],'y':r['label'],'split':split,'color':r['color']})
    order = np.random.default_rng(config['split_protocol']['seed']).permutation(len(rows))
    rows = [rows[i] for i in order]
    for name, subset in [('model_metadata.csv',rows),('discovery_metadata.csv',[r for r in rows if r['split']=='train'])]:
        with (root/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['img_id','img_filename','y','split','color'])
            writer.writeheader();writer.writerows(subset)
    write(root/'partition_map.json',mapping)
    counts={s:sum(r['split']==s for r in rows) for s in ['train','val','id','test']}
    for split in counts:
        if {r['y'] for r in rows if r['split']==split} != {'0','1'}: raise RuntimeError('IRM partition lacks a class')
    manifest={'input_hash':fingerprint,'data_sha256':source_hash,'partition_counts':counts,
        'view_hashes':{f:sha(root/f) for f in ['model_metadata.csv','discovery_metadata.csv','partition_map.json']},
        'source_dataset':str(source),'source_image_count':len(records),'source_inventory_sha256':source_hash,
        'source_metadata_sha256':sha(source/'metadata.csv'),'split_protocol':config['split_protocol'],
        'excluded_training_test_content_overlap':excluded,'discovery_mode':'metadata_assisted_train_partition_only',
        'image_strategy':'hard links to verified original bytes; source read-only',
        'color_encoding':'red/green foreground verified against every RGB pixel',
        'split_isolation':'SHA256 of max(RGB) recovered digit pixels; entire content groups isolated; original OOD retained',
        'ood_environment_role':'independent distribution shift; paired red-green swap is separately constructed on held-out ID',
        'validation_selection':'training environments only; original noisy binary y preserved'}
    write(manifest_path,manifest)
    return root,manifest
