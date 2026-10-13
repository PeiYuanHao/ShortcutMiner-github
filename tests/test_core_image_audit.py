"""Real image training and independent ID/OOD through the five-Agent graph."""
import csv
import json
from pathlib import Path

from PIL import Image
import pytest
import numpy as np

from shortcutminer.audit import build_audit_run_request
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.runner import run, build_local_tool_runtimes
from shortcutminer.runtime.demo import DemoAgentClient


CHECKS = ['association_test', 'task_model_training', 'task_model_evaluation', 'ood_evaluation']


class ImageAuditClient(DemoAgentClient):
    def __init__(self):
        self.called = []
        super().__init__()

    def _factory(self, name):
        original = super()._factory(name)
        def respond(payload, system_prompt):
            self.called.append(name)
            return original(payload, system_prompt)
        return respond

    def _executive(self, payload):
        result = super()._executive(payload)
        for decision, context in zip(result['decisions'], payload['executable_candidate_contexts'], strict=True):
            assert set(CHECKS).issubset(context['allowed_quantitative_checks']), context
            decision['quantitative']['requested_checks'] = CHECKS
            decision['qualitative']['requested_checks'] = list(context['allowed_qualitative_checks'])
        return result


@pytest.mark.parametrize('paired', [False, True, 'color'])
def test_actual_five_agents_train_and_independent_ood(tmp_path, monkeypatch, paired):
    root = tmp_path / 'data'
    root.mkdir()
    rows = []
    for index in range(44):
        split = 'train' if index < 20 else 'val' if index < 28 else 'id' if index < 36 else 'ood'
        label = index % 2
        site = label if split != 'ood' else 1 - label
        path = f'{index}.png'
        Image.new('RGB', (24, 24), (230, 20, 20) if site == 0 else (20, 20, 230)).save(root / path)
        rows.append({'id': str(index), 'path': path, 'label': label, 'split': split, 'site': site})
    for name, selected in [('model.csv', rows), ('discovery.csv', rows[:20])]:
        with (root / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    options = dict(sample_id_column='id', sample_path_column='path', split_column='split',
        model_metadata_file='model.csv', train_split_values=['train'], validation_split_values=['val'],
        test_split_values=['id', 'ood'], evaluation_split_values=['id'],
        model_backend='reference_image_linear', feature_version='reference_image_features_v1',
        max_train_samples=20, max_samples=8, association_variable_mode='categorical_categorical',
        ood_specs={'site': {'id_split_values': ['id'], 'ood_split_values': ['ood']}})
    if paired == 'color':
        options['ood_specs'] = {}
        options['color_ood_specs'] = {'site': {'type': 'global_color', 'mode': 'cycle_rgb'}}
    elif paired:
        from shortcutminer.intervention.segmentation import SegmentationMask
        class FixtureForeground:
            provider_name='fixture_foreground'; provider_version='fixture_v1'
            def __init__(self, spec): self.last_details={}
            def segment(self, *, image_path, mask_path, **limits):
                assert mask_path is None
                mask=np.zeros((24,24),dtype=bool);mask[8:16,8:16]=True
                self.last_details={'oracle_mask_used':False,'fixture_only':True}
                return SegmentationMask(mask=mask,coverage=float(mask.mean()),provider=self.provider_name,version=self.provider_version)
            def close(self): pass
        monkeypatch.setattr('shortcutminer.intervention.local_foreground.LocalForegroundSegmentationProvider',FixtureForeground)
        local={'directory':str(tmp_path/'fixture_model'),'files_sha256':{'config.json':'0'*64}}
        options['ood_specs']={}
        options['background_ood_specs']={'site':{'type':'background_replace','local_foreground':{'sam':local,'grounding_dino':local}}}
    request = build_audit_run_request(AuditRequest(dataset_path=str(root), metadata_file='discovery.csv',
        label_information={'source': 'metadata_column', 'column': 'label'}, execution_options=options),
        run_id='five-agent-image', runs_root=tmp_path / 'runs')
    request = request.model_copy(update={'qualitative_config': {'preparation': {'max_evidence_records': 8}},
        'executive_config': {'audit_options': {
        'required_quantitative_checks': CHECKS, 'max_samples_per_task': 8}},
        'metadata': {**request.metadata, 'promotion_policy': {
            'policy_version': 'fixture-v1', 'calibration_status': 'test_only', 'minimum_samples': 4,
            'metric_rules': [{'experiment_type': 'statistical_check', 'metric': 'effect_size',
                       'operator': 'gt', 'threshold': 0.5}]}}})
    client = ImageAuditClient()
    runtimes = build_local_tool_runtimes(run_id=request.run_id, dataset_root=root,
        runs_root=request.runs_root, max_tool_calls_per_run=80, max_tool_calls_per_task=30)
    result = run(request, agent_client=client, tool_runtimes=runtimes)
    assert result.status == 'COMPLETED'
    assert set(client.called) == {'identify', 'executive', 'quantitative', 'qualitative', 'summarize'}
    directory = Path(request.runs_root) / request.run_id
    assert (directory / 'qualitative/TASK-C001-QL/inspection.json').is_file()
    assert (directory / 'qualitative/TASK-C001-QL/stage2/inspection.json').is_file()
    qualitative_input = json.loads((directory / 'qualitative/TASK-C001-QL/input.json').read_text())
    assert qualitative_input['request']['preparation']['max_evidence_records'] == 8
    assert len(qualitative_input['evidence_context']['selected_samples']) == 8
    results = [json.loads(p.read_text()) for p in directory.rglob('*.json') if 'tool' in str(p)]
    successful = [r for r in results if isinstance(r, dict) and r.get('status') == 'success' and r.get('tool_name') == 'run_experiment']
    kinds = {r.get('experiment_type') or r.get('output', {}).get('experiment_type') for r in successful}
    assert set(['statistical_check', 'task_model_training', 'task_model_evaluation', 'ood_evaluation']).issubset(kinds), kinds
    if paired:
        report=json.loads(next(directory.glob('interventions/**/intervention_manifest.json')).read_text())
        assert report['comparison_mode']=='paired' and report['usable_pair_count']==8
        assert report['failed_pair_count']==0
        if paired == 'color':
            assert report['segmentation_provider'] is None
            for pair in report['pairs']:
                before=np.array(Image.open(root/pair['original_reference']))
                after=np.array(Image.open(directory/pair['counterfactual_reference']))
                assert np.array_equal(after,before[:,:,[2,0,1]])
                assert np.array_equal(after.min(axis=2),before.min(axis=2))
        else:
            assert report['segmentation_provider']=='fixture_foreground'
            for pair in report['pairs']:
                assert pair['quality']['foreground_change_fraction']==0
                assert (directory/pair['mask_reference']).is_file()
        measurement=json.loads(next(directory.glob('interventions/**/measurement.json')).read_text())
        assert len(measurement['rows'])==8
        assert json.loads((directory/'summarize/result.json').read_text())['source']=='summarize'
    else:
        ood = next(directory.glob('models/ood/**/*.json'))
        report = json.loads(ood.read_text())
        assert report['comparison_mode'] == 'independent'
        assert report['profiles'][0]['id_count'] == report['profiles'][0]['ood_count'] == 8
    model = next(json.loads(p.read_text()) for p in directory.glob('models/**/*.json')
                 if json.loads(p.read_text()).get('model_kind') == 'reference_image_linear')
    assert not set(model['training']['train_sample_ids']) & set(model['training']['test_sample_ids'])
