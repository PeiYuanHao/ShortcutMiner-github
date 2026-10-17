"""Sequential GPU-safe E4 runner with live progress, fail-closed source freeze."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write
from experiments.e4_ablation.scripts.collect_results import collect
from shortcutminer.experiment.waterbirds_paper import source_identity
from shortcutminer.experiment.autonomous_audit import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--retry-failed',action='store_true');args=p.parse_args()
    study=args.study.resolve();protocol=json.loads((study/'study.json').read_text(encoding='utf-8'))
    frozen=json.loads((study/'source-identity.json').read_text(encoding='utf-8'))
    if source_identity(require_clean=False)['source_tree_sha256']!=frozen['source']['source_tree_sha256']:
        raise ValueError('Study source changed; never continue a frozen matrix with edited code')
    if sha(study/'profile.json')!=protocol['profile_sha256']:raise ValueError('Profile changed')
    lock=study/'matrix.lock'
    descriptor=None;process=None
    try:
        descriptor=lock.open('x',encoding='utf-8');descriptor.write(str(__import__('os').getpid()));descriptor.flush()
        for arm in protocol['arms']:
            root=Path(arm['directory']);root.mkdir(parents=True,exist_ok=True)
            status_file=root/'arm-status.json'
            status=json.loads(status_file.read_text(encoding='utf-8')).get('status') if status_file.exists() else 'PENDING'
            if status=='COMPLETED' or status=='FAILED' and not args.retry_failed:continue
            if source_identity(require_clean=False)['source_tree_sha256']!=frozen['source']['source_tree_sha256']:
                raise ValueError('Source changed while matrix was running; stopping before next arm')
            spec=protocol['inputs'][arm['case']]
            # Enforce equal frozen public inputs immediately before every arm.
            for mode in ('discovery','training'):
                base=Path(spec[mode]);expected=[r for r in spec['rows'] if mode=='training' or r['split']=='train']
                import csv
                with (base/'metadata.csv').open(encoding='utf-8',newline='') as f:actual=list(csv.DictReader(f))
                if actual!=expected:raise ValueError('Public metadata changed')
                for row in expected:
                    if sha(base/row['img_filename'])!=spec['file_hashes'][row['img_filename']]:raise ValueError('Public input changed')
            command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('run_arm.py')),
                '--variant',arm['variant'],'--input',spec['discovery'],'--training-input',spec['training'],
                '--profile',str(study/'profile.json'),'--runs-root',str(root),'--run-id',arm['run_id'],'--task',spec['task']]
            if protocol.get('protocol_revision'):
                command += ['--protocol-revision', protocol['protocol_revision']]
            print(json.dumps({'event':'ARM_STARTED','case':arm['case'],'variant':arm['variant'],'utc':datetime.now(timezone.utc).isoformat()}),flush=True)
            with (root/'console.log').open('a',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write(study/'current-process.json',{'pid':process.pid,'case':arm['case'],'variant':arm['variant'],'directory':str(root)})
                while process.poll() is None:
                    collect(study);time.sleep(15)
            code=process.returncode
            if code and (not status_file.exists() or json.loads(status_file.read_text())['status']=='RUNNING'):
                write(status_file,{'status':'FAILED','error_type':'ProcessExited','exit_code':code})
            collect(study)
            print(json.dumps({'event':'ARM_FINISHED','case':arm['case'],'variant':arm['variant'],'exit_code':code}),flush=True)
        write(study/'matrix-status.json',{'status':'FINISHED','utc':datetime.now(timezone.utc).isoformat()})
        collect(study)
        if protocol.get('retained_arms'):
            from experiments.e4_ablation.scripts.consolidate_reduced_study import consolidate
            consolidate(study)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
        if descriptor is not None:descriptor.close();lock.unlink(missing_ok=True)


if __name__=='__main__':main()
