"""Training-only engineering throughput check; no paper-result checkpoint produced."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write
from shortcutminer.task_models.torchvision_matrix import TorchvisionResNetMatrix


def main():
    study=ROOT/'results_eswa_single';target=study/'engineering/training-throughput.json'
    if target.exists():raise SystemExit('Existing benchmark preserved')
    case=next(c for c in json.loads((study/'DEVELOPMENT_INPUTS.json').read_text())['cases'] if c['case_id']=='dev-03')
    root=Path(case['training'])
    with (root/'metadata.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    rows=[r for r in rows if r['split']=='train']
    profile=json.loads((study/'protocol/development-profile.json').read_text(encoding='utf-8-sig'))
    options=profile['execution_options']['training_options']
    adapter=TorchvisionResNetMatrix('engineering-throughput-only',options)
    torch=adapter.torch;torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    adapter.classes_=np.array(['0','1']);model=adapter._build('resnet18')
    optimizer=torch.optim.SGD(model.parameters(),lr=options['learning_rate'],
        momentum=options['momentum'],weight_decay=options['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=options['amp'])
    paths=[root/r['img_filename'] for r in rows];labels=[int(r['y']) for r in rows]
    model.train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic();count=0
    for images,truth in adapter._loader(paths,labels,True,42):
        images=images.cuda(non_blocking=True);truth=truth.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',enabled=options['amp']):loss=torch.nn.functional.cross_entropy(model(images),truth)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite benchmark loss')
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();count+=len(truth)
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED',
        'scope':'One training-data pass for engineering throughput only; no model checkpoint, heldout evaluation or scientific result.',
        'images':count,'wall_seconds':elapsed,'images_per_second':count/elapsed,
        'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
        'architecture':'resnet18','training_options':options,
        'caveat':'Includes Windows worker startup; extrapolation to full training is approximate and excludes validation and checkpoint I/O.'}
    write(target,record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
