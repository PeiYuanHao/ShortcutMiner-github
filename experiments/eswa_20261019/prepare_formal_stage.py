"""Prepare the pre-outcome reduced public inputs; no Agent execution or outcomes."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.waterbirds_paper import sha


def main():
    study=ROOT/'results_eswa_single'
    if not (study/'engineering/cached-training-publication-repair-02/REPAIR_RECEIPT.json').exists():raise ValueError('Engineering repair not verified')
    receipt=json.loads((study/'engineering/cached-training-publication-repair-02/REPAIR_RECEIPT.json').read_text())
    if receipt['status']!='success':raise ValueError('Engineering repair failed')
    target=study/'private/FORMAL_REDUCED_MATRIX.json'
    if target.exists() or (study/'FORMAL_PUBLIC_INPUTS.json').exists():raise ValueError('Formal preparation already scheduled; inspect before resuming')
    original=Path('D:/PeiYuan Hao/桌面/ShortcutMiner/ESWA_five_day_matrix_private_2026-10-18.json')
    matrix=json.loads(original.read_text(encoding='utf-8-sig'))
    matrix['tasks']=[task for task in matrix['tasks'] if task['label_cue_match_rho']!=0.75]
    if len(matrix['tasks'])!=7:raise ValueError('Expected predeclared seven-task reduction')
    matrix['status']='PRE_OUTCOME_INPUT_PREPARATION_ONLY'
    write(target,matrix)
    generator=ROOT/'experiments/eswa_20261019/prepare_inputs.py'
    write(study/'protocol/FORMAL_INPUT_FREEZE.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),
        'source_matrix_sha256':sha(original),'reduced_matrix_sha256':sha(target),'generator_sha256':sha(generator),
        'generator_source':generator.read_text(encoding='utf-8'),'selection':'Predeclared remove all rho=.75; no formal outcomes exist.',
        'formal_runner_frozen':False,'formal_collection_started':False})
    subprocess.run([sys.executable,'-X','utf8',str(generator),'--output',str(study),'--matrix',str(target)],cwd=ROOT,check=True)


if __name__=='__main__':main()
