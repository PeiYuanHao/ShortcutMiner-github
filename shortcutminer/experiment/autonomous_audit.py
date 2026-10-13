"""Candidate-bound model inputs, opened only after training construction is frozen."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
from pydantic import TypeAdapter
from shortcutminer.protocol.identify import SafeRelativePath

from shortcutminer.experiment.autonomous_construction import _hash
from shortcutminer.inspection.image import inspect_image_file, flatten_image_descriptors, load_rgb_image_file


def public_rows(root):
    with (Path(root)/'metadata.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['img_id','img_filename','y','split']:
            raise ValueError('Autonomous public intake requires only img_id,img_filename,y,split')
        rows=list(reader)
    if not rows or len(rows)>100000 or len({r['img_id'] for r in rows})!=len(rows):
        raise ValueError('Empty, oversized or duplicate-ID metadata')
    for row in rows:
        TypeAdapter(SafeRelativePath).validate_python(row['img_filename'])
    return rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_autonomous_view(receipt,discovery_root,training_root,view):
    """Bind only the frozen descriptor. Do not create an evaluation OOD beforehand."""
    if receipt.get('receipt_hash')!=_hash({k:v for k,v in receipt.items() if k!='receipt_hash'}):
        raise ValueError('Construction receipt identity mismatch')
    if receipt.get('status')!='selected' or receipt.get('heldout_seen') or receipt.get('truth_seen'):
        raise ValueError('Accepted training-only construction must precede evaluation access')
    discovery_root=Path(discovery_root).resolve(strict=True)
    training_root=Path(training_root).resolve(strict=True)
    view=Path(view).resolve()
    # A derived view must not overwrite or live inside either source directory.
    if any(view==root or view.is_relative_to(root) or root.is_relative_to(view) for root in (discovery_root,training_root)):
        raise ValueError('Derived view must be separate from input datasets')
    if view.exists():
        saved=json.loads((view/'binding_manifest.json').read_text(encoding='utf-8'))
        if saved['construction_receipt_hash']!=receipt['receipt_hash']:
            raise ValueError('Frozen binding identity changed')
        for root,hashes in ((view,saved['file_hashes']),(training_root,saved['source_hashes'])):
            for ref,digest in hashes.items():
                path=(root/ref).resolve(strict=True);path.relative_to(root)
                if sha(path)!=digest:raise ValueError('Frozen view/source changed')
        return saved
    train_rows=public_rows(discovery_root)
    if sha(discovery_root/'metadata.csv')!=receipt['construction_metadata_sha256']:
        raise ValueError('Construction discovery metadata changed')
    if any(r['split']!='train' for r in train_rows):raise ValueError('Discovery view must contain training only')
    rows=public_rows(training_root)
    if set(r['split'] for r in rows)!={'train','val','id'}:
        raise ValueError('Model input requires explicit train/val/id partitions, without prebuilt OOD')
    original={r['img_id']:r for r in train_rows}
    actual={r['img_id']:r for r in rows if r['split']=='train'}
    if actual!=original:raise ValueError('Discovery and model training metadata differ')
    proposal=receipt['proposal'];key=proposal['descriptor_key']
    field='visual_measurement_'+proposal['candidate_id'].lower()
    source_hashes={'metadata.csv':sha(training_root/'metadata.csv')}
    groups={s:set() for s in ('train','val','id')}
    prepared=[]
    # Validate every binding before publishing files. Evaluation pixels open here, after freeze.
    for row in rows:
        source=(training_root/row['img_filename']).resolve(strict=True);source.relative_to(training_root)
        if row['split']=='train':
            other=(discovery_root/row['img_filename']).resolve(strict=True);other.relative_to(discovery_root)
            if sha(source)!=sha(other) or receipt['training_source_hashes'].get(row['img_filename'],sha(source))!=sha(source):
                raise ValueError('Discovery/calibration training pixels changed')
        image=load_rgb_image_file(source)
        groups[row['split']].add(hashlib.sha256(image.rgb.tobytes()).hexdigest())
        inspected=inspect_image_file(source,requested_descriptors=['dimensions','intensity','rgb_stats','saturation','entropy','edge_density','spatial_stats'],max_file_bytes=25000000,max_pixels=20000000,max_frames=1)
        values=flatten_image_descriptors(inspected.descriptors,file_size_bytes=inspected.byte_size)
        if key not in values:raise ValueError('Frozen descriptor unavailable in authoritative image')
        prepared.append({**row,field:values[key]})
        source_hashes[row['img_filename']]=sha(source)
    if any(groups[a]&groups[b] for a,b in (('train','val'),('train','id'),('val','id'))):
        raise ValueError('Exact image overlap across source splits')
    view.mkdir(parents=True)
    for row in rows:
        target=view/row['img_filename'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(training_root/row['img_filename'],target)
    for name,subset in (('model_metadata.csv',prepared),('discovery_metadata.csv',[r for r in prepared if r['split']=='train'])):
        with (view/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['img_id','img_filename','y','split',field]);writer.writeheader();writer.writerows(subset)
    result={'construction_receipt_hash':receipt['receipt_hash'],'feature_field':field,
            'proposal_frozen_before_heldout_access':True,'truth_read':False,'external_ood_used':False,
            'source_hashes':source_hashes,'split_counts':{s:sum(r['split']==s for r in rows) for s in groups},
            'independence':'Exact pixel overlap checked; physical/scene independence remains unverified',
            'file_hashes':{p.relative_to(view).as_posix():sha(p) for p in view.rglob('*') if p.is_file()}}
    (view/'binding_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
