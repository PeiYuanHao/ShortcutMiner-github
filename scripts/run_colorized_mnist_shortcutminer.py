"""Run the actual ShortcutMiner workflow on original Colorized MNIST."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shortcutminer.experiment.colorized_mnist import prepare
from shortcutminer.experiment.waterbirds_paper import sha, source_identity, archive_source, write
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.audit import build_audit_run_request, persist_audit_request
from shortcutminer.experiment.loader import load_model_runtime_config
from shortcutminer.protocol.model import ModelRuntimeConfigSet
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runner import run, resume_run, build_local_tool_runtimes
from shortcutminer.runtime.formal_environment import formal_credentials
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient, UrllibStreamingJsonTransport
from shortcutminer.runtime.qualitative_multimodal import QualitativeMultimodalClient
from shortcutminer.inspection.semantic_provider import OpenAICompatibleSemanticVisualInspector
from shortcutminer.tooling.runtime import ToolBudget


def execute(config_path, run_id, *, live, resume=False):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('dataset_format') == 'synthetic_corner_patch':
        from shortcutminer.experiment.synthetic_corner_patch import prepare as prepare_patch
        root, data = prepare_patch(config)
    elif config.get('dataset_format') == 'irm_colored_mnist':
        from shortcutminer.experiment.irm_colored_mnist import prepare as prepare_irm
        root, data = prepare_irm(config)
    else:
        root, data = prepare(config)
    if not live:
        print(json.dumps({'status': 'PREPARED', **data}, ensure_ascii=False, indent=2))
        return
    counts = data['partition_counts']
    directory = ROOT / 'runs' / run_id
    if resume:
        receipt = json.loads((directory / 'core_execution_receipt.json').read_text(encoding='utf-8'))
        state = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        if state['status'] == 'COMPLETED' or sha(config_path) != receipt['config_sha256'] or data != receipt['data_view']:
            raise RuntimeError('Completed run or changed resume identity')
    with formal_credentials():
        text = load_model_runtime_config(config['text_model_config'])
        vlm = load_model_runtime_config(config['vlm_model_config']).model_copy(update={'max_tokens': 8192})
        if not resume:
            audit = AuditRequest(dataset_path=str(root), metadata_file='discovery_metadata.csv',
                label_information={'source': 'metadata_column', 'column': 'y'}, intake_budget={'max_entries': 70000},
                dataset_description=config.get('dataset_description', 'Original RGB handwritten-digit classification with directory labels 0 through 9. Discovery sees the training partition only. Color metadata is verified against RGB pixels. Held-out ID and validation are isolated by digit-content hashes. Caller-frozen channel cycling can construct paired color interventions while preserving recovered digit intensity. No expected shortcut or non-shortcut verdict is provided.'),
                execution_options={**config['execution_options'], 'max_train_samples': counts['train'], 'max_samples': counts['test'] + counts['id']})
            request = build_audit_run_request(audit, runs_root=str(ROOT / 'runs'), run_id=run_id,
                model_runtime=ModelRuntimeConfigSet(shared=text, per_agent={'qualitative': vlm}))
            request = RunRequest.model_validate({**request.model_dump(mode='json'),
                'identify_config': {'budget': {'max_candidates': 5, 'max_samples': 20, 'max_dataset_entries': 70000, 'max_metadata_rows': 100000},
                    'visual_discovery': {'enabled': True, 'sample_id_column': 'img_id', 'sample_path_column': 'img_filename',
                        'split_column': 'split', 'selected_split': 'train', 'max_samples': 20, 'semantic_enabled': False}},
                'executive_config': {'audit_options': {'validation_depth': 'deep', 'max_samples_per_task': counts['test'],
                    'required_quantitative_checks': ['association_test', 'task_model_training', 'task_model_evaluation', 'ood_evaluation'],
                    **config.get('executive_audit_options', {})}},
                'qualitative_config': {'preparation': {'max_evidence_records': 20, 'max_contact_sheet_samples': 20,
                    'sampling_strategy': 'group_balanced', 'group_columns': ['y']}},
                'metadata': {**request.metadata, 'promotion_policy': config['promotion_policy'],
                    'paper_protocol': config['paper_protocol'], 'data_view_input_hash': data['input_hash'],
                    'source_policy': 'user_authorized_current_source_snapshot'}})
            persist_audit_request(request)
            source = source_identity(require_clean=False)
            write(directory / 'core_execution_receipt.json', {'source': source,
                'source_archive_sha256': archive_source(directory / 'source_snapshot.zip', source['source_tree_sha256']),
                'config_sha256': sha(config_path), 'data_view': data,
                    'scope': 'actual_shortcutminer_workflow_' + config.get('dataset_format', 'original_colorized_mnist'),
                'training_from_baseline_checkpoints': False, 'data_recolored_before_training': False})
        else:
            index = len(list(directory.glob('resume_execution_receipt_*.json'))) + 1
            stem = f'resume_{index:02d}'
            source = source_identity(require_clean=False)
            weights = {p.relative_to(directory).as_posix(): sha(p) for p in directory.glob('models/task/**/*.npz')}
            write(directory / f'resume_execution_receipt_{index:02d}.json', {'source': source,
                'source_archive': stem + '_source_snapshot.zip',
                'source_archive_sha256': archive_source(directory / (stem + '_source_snapshot.zip'), source['source_tree_sha256']),
                'config_sha256': receipt['config_sha256'], 'existing_weights_sha256': weights,
                'created_at': datetime.now(timezone.utc).isoformat()})
        tools = build_local_tool_runtimes(run_id=run_id, dataset_root=root, runs_root=ROOT / 'runs',
            restore_persisted_task_results=resume, max_tool_calls_per_run=150, max_tool_calls_per_task=40)
        for runtime in tools.values():
            runtime.budget = ToolBudget(max_tool_calls=40, timeout_seconds=300,
                experiment_timeout_seconds={'task_model_training': None, 'task_model_evaluation': 1800, 'ood_evaluation': 1800},
                max_selected_samples=counts['test'])
        text_client = OpenAICompatibleClient(text, runs_root=ROOT / 'runs')
        vlm_client = OpenAICompatibleClient(vlm, runs_root=ROOT / 'runs', transport=UrllibStreamingJsonTransport())
        clients = {'agent_client': text_client, 'qualitative_client': QualitativeMultimodalClient(vlm_client, root, max_images=20, contact_sheet=True, runs_root=ROOT/'runs'),
            'tool_runtimes': tools, 'semantic_inspector': OpenAICompatibleSemanticVisualInspector(vlm_client)}
        if resume:
            resume_run(run_id, runs_root=ROOT / 'runs', **clients)
            for ref, expected in weights.items():
                if sha(directory / ref) != expected:
                    raise RuntimeError('Existing training weights changed on resume')
        else:
            run(request, **clients)
        print(json.dumps({'run_id': run_id, 'status': json.loads((directory / 'run.json').read_text())['status']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/colored-mnist-irm-shortcutminer-memory-v2.json')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.resume and (not args.live or not args.run_id):
        parser.error('--resume requires --live and an existing --run-id')
    execute(args.config, args.run_id or 'colorized-mnist-shortcutminer-' + datetime.now().strftime('%Y%m%d-%H%M%S'), live=args.live, resume=args.resume)
