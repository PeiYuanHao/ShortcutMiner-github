import numpy as np
import pytest
from PIL import Image

from shortcutminer.experiment.colorized_mnist import inspect_source, partition
from shortcutminer.intervention.engine import InterventionEngine
from shortcutminer.intervention.protocol import GlobalColorTransform
from shortcutminer.experiment.execution_contract import _authorize_transformation, UnauthorizedScientificField
from test_quantitative_finalizer import _request


def test_tabular_transport_preserves_every_record_and_exact_nested_value():
    import json
    from pathlib import Path
    from shortcutminer.runtime.qualitative_multimodal import _tabular_records
    records = [{'sample_id': str(i), 'descriptor': {'red': i / 7, 'empty': {}},
                'warnings': [], 'available': True} for i in range(20)]
    table = json.loads(json.dumps(_tabular_records(records)))
    restored = []
    for row in table['rows']:
        record = {}
        for path, value in zip(table['columns'], row):
            current = record
            for key in path[:-1]:
                current = current.setdefault(key, {})
            current[path[-1]] = value
        restored.append(record)
    assert restored == records
    assert _tabular_records([{'a': 1}, {'b': 2}]) == [{'a': 1}, {'b': 2}]


@pytest.mark.parametrize('complete', [True, False])
def test_stream_transport_assembles_usage_and_rejects_partial_output(monkeypatch, complete):
    import json
    from io import BytesIO
    from shortcutminer.runtime import openai_compatible as runtime
    chunks = [
        {'choices': [{'index': 0, 'delta': {'content': '{"ok":'}, 'finish_reason': None}]},
        {'choices': [{'index': 0, 'delta': {'content': 'true}'}, 'finish_reason': 'stop'}]},
        {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14}},
    ]
    raw = b''.join(b'data: ' + json.dumps(c).encode() + b'\n\n' for c in chunks)
    if complete:
        raw += b'data: [DONE]\n\n'
    class Response(BytesIO):
        headers = {'Content-Type': 'text/event-stream; charset=utf-8'}
    def open_request(request, timeout):
        assert timeout == 60
        assert json.loads(request.data)['stream'] is True
        return Response(raw)
    monkeypatch.setattr(runtime, 'urlopen', open_request)
    transport = runtime.UrllibStreamingJsonTransport()
    if not complete:
        with pytest.raises(runtime.ProviderResponseJSONError, match='before completion'):
            transport.post_json('https://example.test', headers={}, body={'stream': True}, timeout=60)
    else:
        result = transport.post_json('https://example.test', headers={}, body={'stream': True}, timeout=60)
        assert json.loads(result['choices'][0]['message']['content']) == {'ok': True}
        assert result['choices'][0]['finish_reason'] == 'stop'
        assert result['usage']['total_tokens'] == 14


@pytest.mark.parametrize('rgb', [(255,0,0),(0,255,0),(0,0,255)])
def test_cycle_changes_every_palette_and_preserves_exact_digit_intensity(tmp_path, rgb):
    intensity = np.arange(28*28, dtype=np.uint16).reshape(28,28).astype(np.uint8)
    before = np.maximum(intensity[:,:,None], np.array(rgb,dtype=np.uint8))
    before[0,0] = rgb
    source = tmp_path/'source.png'; target = tmp_path/'modified.png'
    Image.fromarray(before).save(source)
    quality=InterventionEngine().apply(source_path=source,output_path=target,
        transformation=GlobalColorTransform(type='global_color',mode='cycle_rgb'))
    after=np.array(Image.open(target))
    assert quality.same_dimensions and quality.pixel_change_fraction>0
    assert np.array_equal(after,before[:,:,[2,0,1]])
    assert np.array_equal(after.min(axis=2),before.min(axis=2))
    assert not np.array_equal(after[0,0],before[0,0])


def test_color_recipe_cannot_be_changed_by_provider(tmp_path):
    recipe=GlobalColorTransform(type='global_color',mode='cycle_rgb')
    resources=_request(tmp_path).execution_resources.model_copy(update={'color_ood_specs':{'site':recipe}})
    assert _authorize_transformation(recipe,resources,tmp_path)==recipe
    with pytest.raises(UnauthorizedScientificField):
        _authorize_transformation(recipe.model_copy(update={'mode':'grayscale'}),resources,tmp_path)


def test_digit_content_groups_do_not_leak_and_test_overlap_is_excluded():
    records=[{'reference':f'training/{i}.png','original_split':'training',
        'digit_intensity_sha256':f'group-{i}','label':str(i%10)} for i in range(200)]
    records.extend([{'reference':'training/duplicate.png','original_split':'training',
        'digit_intensity_sha256':'group-1','label':'1'},
        {'reference':'testing/overlap.png','original_split':'testing','digit_intensity_sha256':'group-2','label':'2'}])
    assigned,excluded=partition(records,{'seed':42,'id_fraction':.1,'validation_fraction_of_remaining':.2})
    assert excluded==[records[2]] and records[2]['reference'] not in assigned
    assert assigned['training/1.png']==assigned['training/duplicate.png']
    grouped={}
    for record in records:
        if record['reference'] in assigned:
            grouped.setdefault(record['digit_intensity_sha256'],set()).add(assigned[record['reference']])
    assert all(len(splits)==1 for splits in grouped.values())


def test_pixels_must_match_declared_color(tmp_path):
    for split in ['training','testing']:
        (tmp_path/split/'0').mkdir(parents=True)
        Image.new('RGB',(28,28),(0,0,255)).save(tmp_path/split/'0'/'red_1.png')
    with pytest.raises(RuntimeError,match='Pixels disagree'):
        inspect_source(tmp_path)


def test_digit_augmentation_does_not_crop_or_flip(tmp_path):
    from shortcutminer.task_models.torchvision_matrix import PathImages
    sample=tmp_path/'digit.png';Image.new('RGB',(28,28),(255,0,0)).save(sample)
    train=PathImages([sample],[0],True,'digit_preserving')
    evaluate=PathImages([sample],[0],False,'digit_preserving')
    assert [type(t).__name__ for t in train.transform.transforms]==['Resize','ToTensor','Normalize']
    assert np.array_equal(train[0][0].numpy(),evaluate[0][0].numpy())


def test_twenty_image_transport_budget_matches_digit_inspection(tmp_path):
    from types import SimpleNamespace
    from shortcutminer.runtime.qualitative_multimodal import QualitativeMultimodalClient
    Image.new('RGB',(28,28),(255,0,0)).save(tmp_path/'sample.png')
    class Client:
        config=SimpleNamespace()
        def invoke_multimodal(self,**kwargs):
            assert len(kwargs['image_inputs'])==20
            return 'submitted'
    payload={'request':{'run_id':'digit-fixture','task':{'inputs':{'dataset_reference':str(tmp_path)}}},
        'evidence_context':{'selected_samples':[{'sample_available':True,'media_type':'image',
            'inspection_status':'completed','sample_reference':'sample.png','sample_id':str(i),'group':str(i%10)} for i in range(20)]}}
    client=QualitativeMultimodalClient(Client(),tmp_path,max_images=20)
    assert client.invoke_agent_validated('qualitative',payload,system_prompt='',validator=lambda x:x)=='submitted'
    payload['evidence_context']['selected_samples'].append(payload['evidence_context']['selected_samples'][0])
    with pytest.raises(ValueError,match='budget exceeded'):
        client.invoke_agent_validated('qualitative',payload,system_prompt='',validator=lambda x:x)


def test_contact_sheet_keeps_all_samples_and_records_actual_single_image(tmp_path):
    import base64
    import hashlib
    import json
    from io import BytesIO
    from types import SimpleNamespace
    from shortcutminer.runtime.qualitative_multimodal import QualitativeMultimodalClient
    original=np.maximum(np.arange(28*28,dtype=np.uint16).reshape(28,28).astype(np.uint8)[:,:,None],np.array([255,0,0],dtype=np.uint8))
    Image.fromarray(original).save(tmp_path/'sample.png')
    class Client:
        config=SimpleNamespace()
        def invoke_multimodal(self,**kwargs):
            inputs=kwargs['image_inputs'];members=kwargs['payload_for_hash']['contact_sheet_members']
            assert len(inputs)==1 and len(members)==20
            urls=[p for p in kwargs['content_parts'] if p['type']=='image_url'];assert len(urls)==1
            raw=base64.b64decode(urls[0]['image_url']['url'].split(',',1)[1])
            assert hashlib.sha256(raw).hexdigest()==inputs[0].image_sha256
            pixels=np.array(Image.open(BytesIO(raw)))
            for index,member in enumerate(members):
                assert member['sample_id']==str(index) and member['position']==index+1
                x=index%5*144+16;y=index//5*168+24
                assert np.array_equal(pixels[y:y+112:4,x:x+112:4],original)
            assert (tmp_path/'runs/digit-fixture'/inputs[0].sample_reference).read_bytes()==raw
            saved=json.loads((tmp_path/'runs/digit-fixture/qualitative/TASK-C001-QL/submitted_contact_sheet_members.json').read_text())
            assert saved==members
            return 'submitted'
    payload={'request':{'run_id':'digit-fixture','task':{'task_id':'TASK-C001-QL','inputs':{'dataset_reference':str(tmp_path)}}},
        'evidence_context':{'selected_samples':[{'sample_available':True,'media_type':'image',
            'inspection_status':'completed','sample_reference':'sample.png','sample_id':str(i),'group':str(i%10)} for i in range(20)]}}
    with pytest.raises(ValueError,match='explicit runs root'):
        QualitativeMultimodalClient(Client(),tmp_path,max_images=20,contact_sheet=True)
    client=QualitativeMultimodalClient(Client(),tmp_path,max_images=20,contact_sheet=True,runs_root=tmp_path/'runs')
    assert client.invoke_agent_validated('qualitative',payload,system_prompt='',validator=lambda x:x)=='submitted'
