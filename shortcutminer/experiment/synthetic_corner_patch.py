"""Read-only image/label adapter; construction truth excluded from discovery."""
import csv
import hashlib
import os
from pathlib import Path
import numpy as np
from PIL import Image
from shortcutminer.experiment.waterbirds_paper import sha, identity, write
from shortcutminer.intervention.corner_patch import invariant_content, locate, CORNERS

def prepare(config):
    source=Path(config['dataset_root']).resolve(strict=True); records=[]
    split_map={'train':'train','val':'val','test_id':'id','test_ood':'test'}
    with (source/'metadata.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            reference=Path(row['relative_path']).as_posix(); path=(source/reference).resolve(strict=True);path.relative_to(source)
            if row['split'] not in split_map or row['label'] not in {'0','1','2','3'}: raise ValueError('Invalid source split or label')
            with Image.open(path) as image:
                if image.mode!='RGB' or image.size!=(32,32): raise ValueError('Expected original RGB 32x32 image')
                rgb=np.array(image)
            size=int(row['patch_size'])
            if size!=4: raise ValueError('Unexpected source patch geometry')
            corner=CORNERS[locate(rgb,size,[255,0,0])]
            if corner!=row['patch_corner']: raise ValueError('Public patch metadata disagrees with pixels')
            records.append({'sample_id':row['split']+'-'+path.stem,'reference':reference,'label':row['label'],
                'source_split':row['split'],'execution_split':split_map[row['split']],'patch_corner':corner,
                'file_sha256':sha(path),'protected_content_sha256':hashlib.sha256(invariant_content(rgb,size).tobytes()).hexdigest()})
    if len(records)!=24000 or len({r['sample_id'] for r in records})!=len(records): raise ValueError('Unexpected source inventory')
    groups={s:{r['protected_content_sha256'] for r in records if r['execution_split']==s} for s in split_map.values()}
    for i,s in enumerate(groups):
        for t in list(groups)[i+1:]:
            if groups[s]&groups[t]: raise ValueError('Protected image content crosses train/validation/ID/OOD partitions')
    fingerprint=identity(records);root=Path(config['derived_dataset_root']).resolve();manifest_path=root/'input_manifest.json'
    if manifest_path.is_file():
        existing=__import__('json').loads(manifest_path.read_text(encoding='utf-8'))
        if existing['source_inventory_sha256']!=fingerprint or existing['source_metadata_sha256']!=sha(source/'metadata.csv'):
            raise ValueError('Frozen source identity changed')
        for name,value in existing['view_hashes'].items():
            if sha(root/name)!=value: raise ValueError('Frozen execution metadata changed')
        for r in records:
            if sha(root/'images'/r['reference'])!=r['file_sha256']: raise ValueError('Frozen image changed')
        return root,existing
    root.mkdir(parents=True,exist_ok=True);rows=[]
    for r in records:
        target=root/'images'/r['reference'];target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): os.link(source/r['reference'],target)
        if sha(target)!=r['file_sha256']: raise ValueError('Derived image is not original bytes')
        rows.append({'img_id':r['sample_id'],'img_filename':('images/'+r['reference']),
                     'y':r['label'],'split':r['execution_split'],'patch_corner':r['patch_corner']})
    for name,subset in [('model_metadata.csv',rows),('discovery_metadata.csv',[r for r in rows if r['split']=='train'])]:
        with (root/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(subset)
    write(root/'partition_map.json',records)
    counts={s:sum(r['split']==s for r in rows) for s in groups}
    for s in counts:
        if {r['y'] for r in rows if r['split']==s}!={'0','1','2','3'}: raise ValueError('A split lacks a class')
    manifest={'input_hash':fingerprint,'source_inventory_sha256':fingerprint,'source_metadata_sha256':sha(source/'metadata.csv'),
        'source_dataset':str(source),'source_image_count':len(records),'partition_counts':counts,
        'view_hashes':{name:sha(root/name) for name in ['model_metadata.csv','discovery_metadata.csv','partition_map.json']},
        'discovery_mode':'metadata_assisted_training_only','construction_truth_fields_excluded':['expected_corner','is_aligned','align_prob','original_index','class_name'],
        'source_bytes_and_labels_preserved':True,'split_isolation':'exact RGB content outside all corner patch regions',
        'source_index_note':'Original indices may reuse train/test-local namespaces; pixel content rather than numeric index verifies disjointness',
        'label_preservation_limit':'Clean pixels beneath original patches unavailable; interventions cannot recover them',
        'independent_ood_role':'separate original held-out distribution; not used for recipe or checkpoint selection'}
    write(manifest_path,manifest);return root,manifest
