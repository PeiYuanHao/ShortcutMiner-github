from types import SimpleNamespace
import pytest
from PIL import Image
from shortcutminer.runtime.qualitative_multimodal import QualitativeMultimodalClient


def test_current_task_images_only(tmp_path):
    Image.new('RGB', (900, 900), 'red').save(tmp_path / 'sample.png')
    captured = {}
    def invoke(**kwargs):
        captured.update(kwargs)
        return {'checked': True}
    client = QualitativeMultimodalClient(SimpleNamespace(config=None, invoke_multimodal=invoke), tmp_path)
    payload = {'request': {'run_id': 'images', 'task': {'inputs': {'dataset_reference': str(tmp_path)}}},
        'evidence_context': {'selected_samples': [{'sample_available': True, 'media_type': 'image',
            'inspection_status': 'completed', 'sample_reference': 'sample.png', 'sample_id': 'one', 'group': 'A'},
            {'sample_available': False, 'media_type': 'image', 'inspection_status': 'failed', 'sample_reference': 'missing.png'}]}}
    assert client.invoke_agent_validated('qualitative', payload, system_prompt='Inspect evidence', validator=lambda v: None) == {'checked': True}
    assert captured['agent_name'] == 'qualitative'
    assert len(captured['image_inputs']) == 1
    assert captured['image_inputs'][0].width == 768
    assert 'Return comparisons=[]' in captured['system_prompt']
    payload['evidence_context']['selected_samples'].append({
        'sample_available':True,'media_type':'image','inspection_status':'completed',
        'sample_reference':'sample.png','sample_id':'two','group':'B'})
    client.invoke_agent_validated('qualitative',payload,system_prompt='Inspect evidence',validator=lambda v:None)
    assert captured['system_prompt']=='Inspect evidence'
    payload['request']['task']['inputs']['dataset_reference'] = str(tmp_path.parent)
    with pytest.raises(ValueError, match='root differs'):
        client.invoke_agent_validated('qualitative', payload, system_prompt='', validator=lambda v: None)
