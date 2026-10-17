"""Freeze matched public inputs and the E4 matrix; never open hidden truth."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import VARIANTS,write
from shortcutminer.experiment.autonomous_audit import sha
from shortcutminer.experiment.waterbirds_paper import source_identity,archive_source
from shortcutminer.protocol.identify import SafeRelativePath
from pydantic import TypeAdapter

SOURCES={
    'case01':(Path('E:/datasets/E2-A/e2-case-a-training-v2'),'metadata.csv'),
    'case02':(Path('E:/datasets/E2-B/e2-case-b-input-v4'),'model_metadata.csv'),
    'case03':(Path('E:/datasets/E2-C/e2-case-c-input-v1'),'model_metadata.csv'),
    'case04':(ROOT/'runs/e3-representative-20261014/public-training','metadata.csv'),
}
FIELDS=['img_id','img_filename','y','split']


def prepare(target):
    target=Path(target).resolve()
    if target.exists():raise ValueError('Use a fresh study directory; frozen study already exists')
    inputs={}
    for key,(root,filename) in SOURCES.items():
        with (root/filename).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=FIELDS:raise ValueError('Only the four public fields are authorized')
            rows=list(reader)
        if set(r['split'] for r in rows)!={'train','val','id'}:raise ValueError('Expected train/val/id')
        if len({r['img_id'] for r in rows})!=len(rows):raise ValueError('Duplicate sample ID')
        if len({r['img_filename'] for r in rows})!=len(rows):raise ValueError('Duplicate sample path')
        hashes={}; seen={}; counts=Counter(); records=[]
        for r in rows:
            TypeAdapter(SafeRelativePath).validate_python(r['img_filename'])
            source=(root/r['img_filename']).resolve(strict=True);source.relative_to(root.resolve())
            h=sha(source)
            if h in seen:raise ValueError('Duplicate source content across public rows')
            seen[h]=r['img_id'];counts[r['split']]+=1
            # Fresh neutral copies, never a broad copy of a directory containing truth or reports.
            for mode in ('training','discovery') if r['split']=='train' else ('training',):
                destination=target/'inputs'/key/mode/r['img_filename']
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
                if sha(destination)!=h:raise ValueError('Public copy hash mismatch')
            hashes[r['img_filename']]=h;records.append(r)
        for mode in ('training','discovery'):
            out=target/'inputs'/key/mode/'metadata.csv'
            with out.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
                writer.writerows(r for r in records if mode=='training' or r['split']=='train')
        labels=sorted({r['y'] for r in rows})
        identity=hashlib.sha256(json.dumps({'rows':rows,'images':hashes},sort_keys=True).encode()).hexdigest()
        inputs[key]={'discovery':str(target/'inputs'/key/'discovery'),
            'training':str(target/'inputs'/key/'training'),'split_counts':dict(counts),
            'label_counts':dict(Counter((r['split']+':'+r['y']) for r in rows)),
            'public_input_sha256':identity,'source_metadata_sha256':sha(root/filename),
            'task':f'{len(labels)}-class image classification. Target y is an opaque categorical class ID. Discover candidate dataset shortcuts from the approved training evidence.',
            'file_hashes':hashes,'rows':rows}
    profile=json.loads((ROOT/'configs/experiments/image-audit.example.json').read_text(encoding='utf-8'))
    write(target/'profile.json',profile)
    # Counterbalanced dataset-major order. Full is reused in both comparison tables.
    arms=[]
    for i,key in enumerate(inputs):
        order=list(VARIANTS[i:])+list(VARIANTS[:i])
        for variant in order:
            arms.append({'case':key,'variant':variant,'run_id':key+'-'+variant.replace('_','-'),
                'directory':str(target/'arms'/key/variant),'status':'PENDING'})
    protocol={'schema_version':'e4-v1','audit_repetitions':1,'audit_seed':42,
        'audit_repetitions_limitation':'Three task-model seeds are not independent discovery-agent repetitions; no between-audit error bars.',
        'inputs':inputs,'arms':arms,'profile_sha256':sha(target/'profile.json'),
        'budget':{'text_calls':200,'vision_calls':50,'total_tools':150,'max_output_tokens':8192,
                  'max_candidates':5,'max_probe_candidates':3,'max_construction_attempts_per_candidate':3},
        'e3_human_adjudication':None,'truth_access':'Separate post-run researcher scoring only.',
        'independent_discovery':True,'historical_candidate_reports_accessible':False,
        'overall_audit_timeout':None}
    write(target/'study.json',protocol)
    source=source_identity(require_clean=False)
    write(target/'source-identity.json',{'source':source,'archive_sha256':archive_source(target/'source.zip',source['source_tree_sha256'])})
    print(json.dumps({'study':str(target),'arms':len(arms),'split_counts':{k:v['split_counts'] for k,v in inputs.items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    prepare(parser.parse_args().study)
