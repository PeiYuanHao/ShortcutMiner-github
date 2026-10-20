"""Local-only ESWA readiness inventory and two bounded live provider canaries."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.loader import load_model_runtime_config
from shortcutminer.runtime.formal_environment import formal_credentials
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient, ModelRuntimeError
from shortcutminer.protocol.model_runtime import ModelImageInput


def now():
    return datetime.now(timezone.utc).isoformat()


def canary(output: Path, kind: str):
    config = load_model_runtime_config(ROOT / f'configs/models/pilot_{kind}.example.yaml')
    config = config.model_copy(update={
        'timeout': 90.0, 'max_tokens': 256,
        'model_call_budget': config.model_call_budget.model_copy(update={
            'max_total_calls': 1, 'max_calls_per_agent': 1,
            'max_schema_retries': 0, 'max_transport_retries': 0}),
    })
    client = OpenAICompatibleClient(config, runs_root=output / 'provider_canaries')
    start = time.monotonic()
    record = {'kind': kind, 'requested_model': config.model, 'started_utc': now()}
    try:
        if kind == 'text':
            result = client.invoke_agent('identify', {
                'run_id': 'canary-text',
                'instruction': 'Connectivity test only. Return a JSON object with ready=true.'},
                system_prompt='Return only the requested JSON object.')
            record['semantic_check'] = result.get('ready') is True
        else:
            from PIL import Image
            image = Image.new('RGB', (64, 64), (230, 30, 30))
            buffer = io.BytesIO(); image.save(buffer, format='PNG'); content = buffer.getvalue()
            result = client.invoke_multimodal(run_id='canary-vlm',
                payload_for_hash={'run_id': 'canary-vlm', 'purpose': 'synthetic image connectivity'},
                system_prompt='Return a JSON object with dominant_color as a lowercase English word.',
                content_parts=[{'type': 'text', 'text': 'What is the dominant color?'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(content).decode()}}],
                validator=lambda value: None,
                image_inputs=[ModelImageInput(sample_id='synthetic', sample_reference='synthetic.png',
                    image_sha256=hashlib.sha256(content).hexdigest(), mime_type='image/png', width=64, height=64)])
            record['semantic_check'] = str(result.get('dominant_color', '')).lower() == 'red'
        record['status'] = 'PASSED' if record['semantic_check'] else 'UNEXPECTED_CONTENT'
    except ModelRuntimeError as exc:
        record.update(status='FAILED', error_type=type(exc).__name__,
            failure_category=exc.failure.category if exc.failure else None,
            http_status=exc.failure.http_status if exc.failure else None)
    except Exception as exc:
        record.update(status='FAILED', error_type=type(exc).__name__)
    record.update(wall_seconds=round(time.monotonic()-start, 3), finished_utc=now(),
        calls=[{'status': r.status, 'http_status': r.http_status, 'usage': r.usage.model_dump(mode='json')}
               for r in client.invocations])
    write(output / f'canary_{kind}.json', record)
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    start_file = out / 'STUDY_START.json'
    if not start_file.exists():
        started = datetime.now(timezone.utc)
        write(start_file, {'started_utc': started.isoformat(),
            'experiment_deadline_utc': (started + timedelta(hours=120)).isoformat(),
            'timezone': 'Asia/Shanghai', 'stage': 'development_preflight',
            'formal_started': False, 'formal_requires_protocol_freeze': True,
            'scope': '10 tasks x 3 main configurations; 9 fixed-candidate ablations; 1 Pet; <=3 development audits',
            'classifier_scope': 'ResNet18, up to 16 new synthetic profiles; old architectures reported separately',
            'notes': 'All scientific outcomes retained. No private truth is agent-facing. No automatic commits.'})
    import torch
    import torchvision
    inventory = {'created_utc': now(), 'repository': str(ROOT), 'python': platform.python_version(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
        'legacy_roots': [], 'new_results_preexisting': (out/'TASK_STATUS.csv').exists()}
    for folder in sorted((ROOT/'runs').iterdir()):
        if not folder.is_dir() or folder.name in {'knowledge', 'organization'}:
            continue
        weights = list(folder.rglob('best*.npz'))
        summaries = list(folder.rglob('summarize/result.json'))
        inventory['legacy_roots'].append({'path': str(folder), 'best_numeric_checkpoints': len(weights),
            'summaries': len(summaries), 'checkpoint_examples': [str(p) for p in weights[:2]],
            'reuse_status': 'NOT_VERIFIED; bytes, labels, split and training recipe must match'})
    write(out/'READINESS.json', inventory)
    print(json.dumps({'stage': 'inventory_saved', 'gpu': inventory['gpu'],
        'legacy_roots': len(inventory['legacy_roots']),
        'numeric_best_checkpoints': sum(r['best_numeric_checkpoints'] for r in inventory['legacy_roots'])}), flush=True)
    if args.live:
        with formal_credentials():
            records = [canary(out, kind) for kind in ('text', 'vlm')]
        if not all(r['status'] == 'PASSED' for r in records):
            raise SystemExit(2)


if __name__ == '__main__':
    main()
