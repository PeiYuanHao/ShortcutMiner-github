"""Freeze the user-approved two-cell recovery; reference eight existing cells."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.waterbirds_paper import source_identity,archive_source
from shortcutminer.experiment.autonomous_audit import sha

SELECTED=('single_agent','fixed_pipeline','without_qualitative','without_quantitative','without_adaptive_routing')
HISTORICAL_FULL={
    'case01':ROOT/'runs/e2-a-isolated-20261009/case-a-retest-r2',
    'case02':ROOT/'runs/e2-b-v4-isolated-20261009/case-b-v4-r2',
}


def prepare(previous,target,protocol_revision='metadata-contract-v2'):
    previous=Path(previous).resolve();target=Path(target).resolve()
    if target.exists():raise ValueError('A fresh study directory is required')
    old=json.loads((previous/'study.json').read_text(encoding='utf-8'))
    selected=[a for a in old['arms'] if a['case'] in ('case01','case02') and a['variant'] in SELECTED]
    new=[];retained=[]
    for a in selected:
        a=dict(a)
        if a['case']=='case01' and a['variant'] in ('single_agent','fixed_pipeline'):
            a.update(directory=str(target/'arms'/a['case']/a['variant']),status='PENDING')
            new.append(a)
        else:
            p=Path(a['directory'])/'arm-status.json'
            status=json.loads(p.read_text(encoding='utf-8'))
            if status['status']!='COMPLETED':raise ValueError('Reused cell is not complete')
            retained.append({**a,'status_sha256':sha(p),'origin_study':str(previous)})
    assert len(new)==2 and len(retained)==8
    for path in HISTORICAL_FULL.values():
        if not (path/'summarize/result.json').exists():raise ValueError('Historical Full report missing')
    target.mkdir(parents=True)
    shutil.copy2(previous/'profile.json',target/'profile.json')
    if sha(target/'profile.json')!=old['profile_sha256']:raise ValueError('Profile identity mismatch')
    protocol={**old,'schema_version':'e4-reduced-recovery-v1','inputs':{k:old['inputs'][k] for k in ('case01','case02')},
        'arms':new,'retained_arms':retained,'selected_variants':list(SELECTED),
        'historical_full':{k:str(v) for k,v in HISTORICAL_FULL.items()},
        'protocol_revision':protocol_revision,'previous_study':str(previous),
        'scope_decision':'User reduced scope after inspecting execution outcomes: E2-A/B only, five variants, historical Full references; no counterevidence ablation.',
        'compatibility':'Same frozen public inputs/profile/budgets/gates; new baseline source and protocol-guidance revision are disclosed. Historical Full is a reference, not asserted to be an identically configured concurrent control.'}
    write(target/'study.json',protocol)
    source=source_identity(require_clean=False)
    write(target/'source-identity.json',{'source':source,'archive_sha256':archive_source(target/'source.zip',source['source_tree_sha256'])})
    print(json.dumps({'study':str(target),'new_runs':len(new),'reused_runs':len(retained)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--protocol-revision',choices=['metadata-contract-v2','descriptor-codec-v3'],default='metadata-contract-v2')
    a=p.parse_args();prepare(a.previous,a.study,a.protocol_revision)
