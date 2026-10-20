"""Evaluator-only CIFAR source isolation and opaque public audit inputs.

Hidden mechanism parameters and source identities never enter public metadata.
The official batch checksums are verified before loading their NumPy arrays.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.e4_ablation.variants.policies import write

FIELDS = ['img_id', 'img_filename', 'y', 'split']
CHECKSUMS = {
    'data_batch_1': 'c99cafc152244af753f735de768cd75f',
    'data_batch_2': 'd4bba439e000b95fd0a9bffe97cbabec',
    'data_batch_3': '54ebc095f3ab1f0389bbae665268c751',
    'data_batch_4': '634d18415352ddfa80567beed471001a',
    'data_batch_5': '482c414d41f54cd18b22e5b47cb7c3cb',
    'test_batch': '40351d587109b95175f43aff81a1287e',
}


def digest(data): return hashlib.sha256(data).hexdigest()


def csv_write(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({key: r[key] for key in fields} for r in rows)


def sources(source):
    rows = []; seen = set(); manifests = []
    for name, expected in CHECKSUMS.items():
        data = (source/name).read_bytes()
        if hashlib.md5(data).hexdigest() != expected:
            raise ValueError('Official CIFAR checksum mismatch: '+name)
        manifests.append({'name': name, 'md5': expected, 'sha256': digest(data)})
        batch = pickle.loads(data, encoding='bytes')
        for index, (flat, label) in enumerate(zip(batch[b'data'], batch[b'labels'])):
            pixels = flat.reshape(3, 32, 32).transpose(1, 2, 0).copy()
            sha = digest(pixels.tobytes())
            if sha in seen: continue
            seen.add(sha)
            rows.append({'batch': name, 'row': index, 'label': int(label), 'sha256': sha, 'pixels': pixels})
    return rows, manifests


def old_sources():
    excluded = set(); inventories = []
    for path in (Path('E:/datasets/E2-A/e2-private/hidden_truth.csv'),
                 Path('E:/datasets/E2-C/e2_truth/hidden_truth.csv')):
        if not path.exists(): raise ValueError('Historical source truth missing: '+str(path))
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        excluded.update(r['original_sha256'] for r in rows)
        inventories.append({'path': str(path), 'sha256': digest(path.read_bytes()), 'rows': len(rows)})
    return excluded, inventories


def select_pool(rows, classes, counts, seed, excluded):
    rng = random.Random(seed); selected = []; seen = set()
    for y, label in enumerate(classes):
        available = [r for r in rows if r['label'] == label and r['sha256'] not in excluded]
        rng.shuffle(available)
        needed = sum(counts.values())//2
        if len(available) < needed: raise ValueError(f'Not enough untouched sources for label {label}: {len(available)} < {needed}')
        offset = 0
        for split, count in counts.items():
            for row in available[offset:offset+count//2]:
                if row['sha256'] in seen: raise ValueError('Duplicate pristine source')
                seen.add(row['sha256']); selected.append({**row, 'y': str(y), 'split': split})
            offset += count//2
    return selected


def transform(pixels, mechanism, state, geometry):
    arr = pixels.copy(); support = np.zeros((40, 40), dtype=bool)
    # Every condition shares the same canvas and central content placement.
    canvas = np.full((40, 40, 3), 127, dtype=np.uint8); canvas[4:36, 4:36] = arr
    if mechanism == 'border':
        support[:] = True; support[4:36, 4:36] = False
        canvas[support] = (205, 75, 80) if state else (65, 155, 200)
    elif mechanism == 'watermark' and state:
        bitmap = np.array([[1,0,1,0,1,0,1,1,0], [1,0,1,0,1,0,1,0,1],
                           [1,0,1,0,1,0,1,0,1], [1,1,1,1,1,0,1,0,1],
                           [0,1,0,1,0,0,1,0,1]], dtype=bool)
        x,y = geometry['watermark_xy']; h,w = bitmap.shape
        region = canvas[y:y+h,x:x+w]; region[bitmap] = [250, 250, 250]
        support[y:y+h,x:x+w] = bitmap
    elif mechanism == 'position_patch':
        x,y = geometry['patch_xy'][state]
        canvas[y:y+4,x:x+4] = [245, 220, 40]; support[y:y+4,x:x+4] = True
    return canvas, support


def make_case(output, case_id, pool, spec, *, development):
    case = output/'public'/case_id
    if case.exists(): raise ValueError('Public case already exists; do not overwrite '+case_id)
    rng = random.Random(spec['private_generation_seed'])
    geometry = {'watermark_xy': rng.choice([(25,29),(6,29),(25,6),(6,6)]),
                'patch_xy': [(5,5),(31,31)]}
    states = {}
    for split in ('train','val','id'):
        for y in ('0','1'):
            group = [r for r in pool if r['split']==split and r['y']==y]
            rng.shuffle(group)
            rho = spec.get('label_cue_match_rho')
            matched = round(len(group)*(rho if rho is not None else .5))
            for i,row in enumerate(group): states[row['sha256']] = int(y) if i < matched else 1-int(y)
    public = []; truth = []; pixels_seen = set()
    ordered = list(pool); rng.shuffle(ordered)
    for row in ordered:
        opaque = digest((case_id+':'+row['sha256']).encode())[:24]
        filename = 'images/'+opaque+'.png'
        state = states[row['sha256']]
        pixels,support = transform(row['pixels'],spec['mechanism'],state,geometry)
        image_hash = digest(pixels.tobytes())
        if image_hash in pixels_seen: raise ValueError('Transformed duplicate')
        pixels_seen.add(image_hash)
        path = case/'training'/filename; path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pixels).save(path, compress_level=1)
        if row['split']=='train':
            destination=case/'discovery'/filename; destination.parent.mkdir(parents=True,exist_ok=True)
            # Hardlinks save disk space; neither experiment entry may alter public image bytes.
            destination.hardlink_to(path)
        public.append(dict(img_id=opaque,img_filename=filename,y=row['y'],split=row['split']))
        truth.append({'img_id':opaque,'y':row['y'],'split':row['split'],
            'source_batch':row['batch'],'source_row':row['row'],'source_sha256':row['sha256'],
            'state':state,'public_rgb_sha256':image_hash,'public_file_sha256':digest(path.read_bytes()),
            'support_pixels':int(support.sum())})
    csv_write(case/'training/metadata.csv',public)
    csv_write(case/'discovery/metadata.csv',[r for r in public if r['split']=='train'])
    record={'case_id':case_id,'development':development,'spec':spec,'geometry':geometry,
        'split_counts':dict(Counter(r['split'] for r in truth)), 'records':truth,
        'metadata_sha256':digest((case/'training/metadata.csv').read_bytes()),
        'discovery':str(case/'discovery'),'training':str(case/'training'),
        'task':'2-class image classification. Target y is an opaque categorical class ID. Discover candidate dataset shortcuts from approved training evidence.',
        'dependence':'All formal cases reuse this source pool and source splits. They are dependent task conditions, not independent datasets.'}
    write(output/'private'/('development' if development else 'formal')/(case_id+'.json'),record)
    return {key:record[key] for key in ('case_id','development','discovery','training','task','split_counts','metadata_sha256')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--development-only',action='store_true')
    parser.add_argument('--matrix',type=Path)
    args=parser.parse_args(); output=args.output.resolve()
    loaded,manifests=sources(Path('E:/datasets/E1/cifar-10-python/cifar-10-batches-py'))
    excluded,historical=old_sources()
    dev_pool=select_pool(loaded,(0,1),{'train':600,'val':120,'id':400},2026101901,excluded)
    specs=[{'mechanism':'watermark','label_cue_match_rho':.95,'private_generation_seed':991101},
           {'mechanism':'watermark','label_cue_match_rho':.5,'private_generation_seed':991102},
           {'mechanism':'no_artificial_cue','label_cue_match_rho':None,'private_generation_seed':991103}]
    development=[]
    for i,spec in enumerate(specs):
        case_id=f'dev-{i+1:02d}'
        private=output/'private/development'/(case_id+'.json')
        if private.exists():
            old=json.loads(private.read_text()); development.append({k:old[k] for k in ('case_id','development','discovery','training','task','split_counts','metadata_sha256')})
        else: development.append(make_case(output,case_id,dev_pool,spec,development=True))
    write(output/'DEVELOPMENT_INPUTS.json',{'cases':development,'source_files':manifests,
        'classes_private':[0,1],'not_formal_results':True,'historical_exclusion_files':historical})
    if not args.development_only:
        if args.matrix is None: raise ValueError('--matrix required for formal generation')
        # 5,600 sources, disjoint from legacy A/C and development. Official batch identity is retained.
        pool=select_pool(loaded,(3,5),{'train':3000,'val':600,'id':2000},2026101902,excluded)
        assert not {r['sha256'] for r in pool}&{r['sha256'] for r in dev_pool}
        matrix=json.loads(args.matrix.read_text(encoding='utf-8'))
        rng=random.Random(2026101903); tasks=sorted(matrix['tasks'],key=lambda t:t['private_schedule_order'])
        ids=[f'p-{digest(str(rng.getrandbits(128)).encode())[:12]}' for _ in tasks]
        cases=[]
        for case_id,spec in zip(ids,tasks):
            cases.append(make_case(output,case_id,pool,spec,development=False))
        write(output/'FORMAL_PUBLIC_INPUTS.json',{'cases':cases,
            'source_pool_independent_datasets':1,'not_independent_task_conditions':True})
        write(output/'private/FORMAL_SOURCE_ISOLATION.json',{'source_files':manifests,
            'source_sha256':[r['sha256'] for r in pool], 'historical_excluded_sources':len(excluded),
            'overlap_with_legacy':0,'overlap_with_development':0,
            'split_counts':dict(Counter(r['split'] for r in pool)),
            'note':'Source exact deduplication does not establish physical object independence.'})
    print(json.dumps({'status':'prepared','development_cases':len(development),'formal_generated':not args.development_only}),flush=True)


if __name__=='__main__': main()
