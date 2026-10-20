"""Create an honest report and a lossless content-deduplicated local evidence archive."""
import argparse,csv,hashlib,json,re,shutil,sys,time,zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];MAIN=ROOT/'results_eswa_single'

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def report(folder):
    """Render only the supplied local results; the public source embeds no outcomes."""
    s=json.loads((folder/'SUMMARY.json').read_text(encoding='utf-8'))
    lines=['# Local evidence report','',
        'This report is generated from local artifacts. Execution completion is not discovery correctness.','',
        f"Agent audits: {s['agent_audits']}; completed: {s['completed']}; failed retained: {s['failed_retained']}.",'',
        '| Configuration | Scheduled | Completed | Failed | Known tokens |',
        '|---|---:|---:|---:|---:|']
    for item in s['main']:
        lines.append(f"| {item['condition']} | {item['scheduled']} | {item['completed']} | {item['failed_retained']} | {item['known_tokens']} |")
    lines += ['',f"All-attempt known tokens: {s['known_tokens_all_attempts']}. Missing-usage requests: {s['missing_usage_requests']}. Reservation estimates: {s['reserved_unknown_estimates']} (not invoices or certified upper bounds).",'',
        f"Human adjudication: {s['human_adjudication']}. Precision/recall/FPR are unavailable until actual independent truth labels are supplied.",'',
        'Interpret model effects only alongside construction fidelity, source dependence, conditional uncertainty and failures. Evaluator oracle operations are separate from autonomous discovery.','',
        'See the supplied local tables, PENDING_AND_DEFERRED.json, logits and source references. No unpublished outcome is embedded in this public reporting helper.']
    (folder/'LOCAL_EVIDENCE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

RESTORE='''"""Restore local evidence. Usage: python RESTORE.py bundle.zip output_folder"""
import csv,hashlib,io,sys,zipfile
from pathlib import Path,PurePosixPath
archive,destination=Path(sys.argv[1]),Path(sys.argv[2]).resolve()
destination.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for row in csv.DictReader(io.TextIOWrapper(z.open('FILE_MANIFEST.csv'),encoding='utf-8')):
        p=PurePosixPath(row['path'])
        if p.is_absolute() or '..' in p.parts or ':' in str(p):raise ValueError('Unsafe archive path')
        target=destination.joinpath(*p.parts)
        if target.exists():raise FileExistsError(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256()
        with z.open(row['object']) as source,target.open('xb') as out:
            for block in iter(lambda:source.read(1024*1024),b''):h.update(block);out.write(block)
        if h.hexdigest()!=row['sha256']:raise ValueError('Integrity failure: '+str(p))
print('Restored and SHA256-verified all manifest paths.')
'''

def package(folder):
    bundle=folder.parent/(folder.name+'-evidence.zip')
    if bundle.exists():raise FileExistsError(bundle)
    excluded=[];files=[];volatile=('MONITOR','LIVE_STATUS','monitor.lock','extension.queue.lock')
    for root,prefix in [(MAIN,'raw'),(folder,'report')]:
        for path in sorted(root.rglob('*')):
            if not path.is_file():continue
            rel=path.relative_to(root)
            if prefix=='raw' and (rel.parts[0]=='delivery' or (len(rel.parts)==1 and path.name.startswith(volatile))):continue
            if rel.parts[:2]==('engineering','plot-runtime'):continue
            if any(part.lower().startswith('.env') or part.lower() in {'__pycache__','.git'} or part.lower().endswith('.lock') for part in rel.parts):
                excluded.append(str(rel));continue
            files.append((path,prefix+'/'+rel.as_posix()))
    manifest=folder/'FILE_MANIFEST.csv';objects={};rows=[];logical=0
    with manifest.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['path','size','sha256','object']);writer.writeheader()
        for index,(path,name) in enumerate(files):
            digest=sha(path);size=path.stat().st_size;row={'path':name,'size':size,'sha256':digest,'object':'objects/'+digest}
            rows.append(row);writer.writerow(row);objects.setdefault(digest,path);logical+=size
            if index%20000==0:print(json.dumps({'stage':'hashing','files':index,'total':len(files)}),flush=True)
    restore=folder/'RESTORE.py';restore.write_text(RESTORE,encoding='utf-8')
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        archive.write(manifest,'FILE_MANIFEST.csv');archive.write(restore,'RESTORE.py')
        for index,(digest,path) in enumerate(objects.items()):
            archive.write(path,'objects/'+digest)
            if index%10000==0:print(json.dumps({'stage':'packing','objects':index,'total':len(objects)}),flush=True)
    with zipfile.ZipFile(bundle) as archive:
        for index,digest in enumerate(objects):
            h=hashlib.sha256()
            with archive.open('objects/'+digest) as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=digest:raise ValueError('Archive object SHA mismatch')
            if index%10000==0:print(json.dumps({'stage':'verifying','objects':index,'total':len(objects)}),flush=True)
    receipt={'status':'VERIFIED','bundle':str(bundle),'bundle_sha256':sha(bundle),'bundle_bytes':bundle.stat().st_size,
        'logical_files':len(rows),'logical_bytes':logical,'unique_objects':len(objects),'excluded_sensitive_or_ephemeral':excluded,
        'verification':'Every archived data object streamed, CRC-checked and SHA256 matched to its content-addressed name; all logical paths in manifest.',
        'created_utc':datetime.now(timezone.utc).isoformat()}
    (folder/'PACKAGE_VERIFICATION.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--delivery',type=Path,required=True);parser.add_argument('--package',action='store_true');args=parser.parse_args()
    if args.package:package(args.delivery)
    else:report(args.delivery)
