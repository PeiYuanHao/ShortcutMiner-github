"""Offline evidence-boundary regressions for literature-driven construction."""
import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pydantic import ValidationError

from shortcutminer.experiment.autonomous_construction import ConstructionProposal, select_autonomous_construction
from shortcutminer.experiment.luminance_calibration import calibrate_luminance
from shortcutminer.experiment.shortcut_knowledge import load_shortcut_knowledge, hypothesis_coverage_context
from shortcutminer.intervention.engine import _transform_basic
from shortcutminer.intervention.protocol import LuminanceTransform, RotationTransform, FrequencyTransform
from shortcutminer.inspection.image import inspect_image_file, flatten_image_descriptors, DESCRIPTOR_DEFINITIONS
from shortcutminer.tooling.base import ToolExecutionContext
from shortcutminer.tooling.builtins import build_default_tool_registry, EmptyInput


def fixture_data(root):
    root.mkdir(exist_ok=True)
    rows=[]
    for i,level in enumerate([60,80,160,180,250]):
        a=np.full((12,12,3),level,dtype=np.uint8)
        a[2:6,2:6]=level+3
        Image.fromarray(a).save(root/f'{i}.png')
        rows.append({'img_id':str(i),'img_filename':f'{i}.png','y':str(i%2),'split':'train' if i<4 else 'id'})
    pd.DataFrame(rows).to_csv(root/'metadata.csv',index=False)
    return SimpleNamespace(model_metadata_file='metadata.csv',split_column='split',train_split_values=['train'],sample_path_column='img_filename',max_train_samples=100)


def test_training_calibration_ignores_labels_and_heldout_pixels(tmp_path,monkeypatch):
    resources=fixture_data(tmp_path)
    from shortcutminer.experiment import luminance_calibration as module
    load=module.load_rgb_image_file
    opened=[]
    def record(path):
        opened.append(path.name)
        return load(path)
    monkeypatch.setattr(module,'load_rgb_image_file',record)
    choice={'mode':'reflect_mean','strategy':'mid_quantiles'}
    first=calibrate_luminance(choice,resources,tmp_path)
    frame=pd.read_csv(tmp_path/'metadata.csv');frame['y']=1-frame.y;frame.to_csv(tmp_path/'metadata.csv',index=False)
    Image.new('RGB',(12,12),'black').save(tmp_path/'4.png')
    second=calibrate_luminance(choice,resources,tmp_path)
    assert first==second
    assert set(opened)=={'0.png','1.png','2.png','3.png'}
    assert 119 < first['transformation']['pivot_mean'] < 121
    resources.model_metadata_file='../metadata.csv'
    with pytest.raises((ValueError,FileNotFoundError)):calibrate_luminance(choice,resources,tmp_path)


@pytest.mark.parametrize('operation',[
    LuminanceTransform(type='luminance',mode='normalize_mean',target_mean=120),
    LuminanceTransform(type='luminance',mode='reflect_mean',pivot_mean=120),
    RotationTransform(type='rotation',angle_degrees=10),
    FrequencyTransform(type='frequency_filter',mode='low_pass',cutoff=.2),
    FrequencyTransform(type='frequency_filter',mode='high_pass',cutoff=.2),
])
def test_new_renderers_are_deterministic_actual_rgb_operations(operation):
    rng=np.random.default_rng(5)
    a=rng.integers(40,120,(24,24,3),dtype=np.uint8)
    b=_transform_basic(a,operation)
    assert b.shape==a.shape and b.dtype==a.dtype
    assert np.array_equal(b,_transform_basic(a,operation))
    assert not np.array_equal(a,b)
    if operation.type=='luminance':
        old=np.array(Image.fromarray(a).convert('YCbCr'))
        new=np.array(Image.fromarray(b).convert('YCbCr'))
        desired=120 if operation.mode=='normalize_mean' else 240-old[:,:,0].mean()
        assert new[:,:,0].mean()==pytest.approx(desired,abs=2)
        assert np.abs(old[:,:,1:].astype(float)-new[:,:,1:]).mean()<2


def test_luminance_schema_and_proposal_do_not_accept_guessed_pivots():
    with pytest.raises(ValidationError):LuminanceTransform(type='luminance',mode='reflect_mean',pivot_mean=float('nan'))
    with pytest.raises(ValidationError):LuminanceTransform(type='luminance',mode='reflect_mean',target_mean=100)
    with pytest.raises(ValidationError):ConstructionProposal(status='selected',candidate_id='C001',descriptor_key='brightness',transformation={'type':'luminance','mode':'reflect_mean','pivot_mean':100},rationale='test',intended_invariants=['dimensions'],limitations=['unverified labels'])


def test_pixel_luminance_units_and_independent_axis(tmp_path):
    Image.new('RGB',(10,10),(90,90,90)).save(tmp_path/'a.png')
    inspected=inspect_image_file(tmp_path/'a.png',requested_descriptors=['intensity'],max_file_bytes=10000,max_pixels=10000,max_frames=1)
    values=flatten_image_descriptors(inspected.descriptors)
    assert values['luminance_y_mean']==90
    assert values['intensity_mean']==pytest.approx(90/255)
    assert DESCRIPTOR_DEFINITIONS['luminance_y']['definition_version']=='ycbcr_y_v1'
    visual=SimpleNamespace(group_summaries=[SimpleNamespace(group={'y':'0'},decoded_count=3,descriptor_means={'luminance_y_mean':90,'grayscale_entropy':5,'rgb_mean_red':.4})])
    coverage=hypothesis_coverage_context(visual)
    axes={a['axis']:a['observed_groups'] for a in coverage['axes']}
    assert axes['global_photometry'][0]['descriptor_means']=={'luminance_y_mean':90}
    assert 'texture_structure' in axes and 'chromaticity' in axes


class TextClient:
    def __init__(self):self.calls=[]
    def invoke_agent(self,name,payload,**kwargs):
        self.calls.append((name,payload))
        if name=='quantitative':
            return {'status':'selected','candidate_id':'C001','descriptor_key':'brightness','luminance_calibration':{'mode':'normalize_mean','strategy':'pooled_median'},'rationale':'Measured photometric variation','intended_invariants':['dimensions'],'limitations':['Class semantics unverified']}
        assert payload['visual_review']['reviewed_pair_ids']
        assert 'changed_mean_y' in payload['pixel_diagnostics'][0]
        return {'decision':'accept','rationale':'Limited dependence diagnostic','limitations':['Semantic review is fallible']}


class VisionClient:
    def __init__(self,status='supported'):self.calls=[];self.status=status
    def invoke_multimodal(self,**kwargs):
        self.calls.append(kwargs)
        part=kwargs['content_parts'][1]['image_url']['url']
        data=base64.b64decode(part.split(',',1)[1])
        telemetry=kwargs['image_inputs'][0]
        assert hashlib.sha256(data).hexdigest()==telemetry.image_sha256
        image=Image.open(BytesIO(data));assert image.size==(384,640)
        assert not np.array_equal(np.array(image)[:,:184],np.array(image)[:,192:376])
        ids=[r['pair_id'] for r in kwargs['payload_for_hash']['training_pairs']]
        return {'status':self.status,'reviewed_pair_ids':ids,'cue_change_observation':'Exposure changes visible','task_content_observation':'Structure remains visible; labels unverified','artifacts_observation':'Round-trip quantization','limitations':['No semantic ground truth']}


def test_construction_calls_visual_agent_with_actual_pairs_and_freezes_before_evaluation(tmp_path):
    resources=fixture_data(tmp_path/'data');text=TextClient();vision=VisionClient()
    args=dict(run_id='offline-fixture',candidates=[{'candidate_id':'C001','hypothesis':'Photometric dependence'}],observations=[],descriptor_keys=['brightness'],resources=resources,root=tmp_path/'data',receipt_path=tmp_path/'control/receipt.json',task_description='Generic binary classification',max_revisions=0)
    receipt=select_autonomous_construction(text,vision,**args)
    assert receipt['status']=='selected' and not receipt['heldout_seen'] and not receipt['truth_seen']
    assert receipt['transformation']['target_mean']==pytest.approx(120.333333333)
    assert receipt['cue_execution_check']['mode']=='normalize_mean'
    assert receipt['cue_execution_check']['changed_y_mean_range'] < 2
    assert 'diagnostic_design_guidance' in text.calls[0][1]
    assert len(vision.calls)==1 and [n for n,p in text.calls]==['quantitative','executive']
    assert all(p['run_id']=='offline-fixture' for n,p in text.calls)
    assert select_autonomous_construction(text,vision,**args)==receipt
    assert len(vision.calls)==1
    path=tmp_path/'control/preview-0/pair-000-changed.png';path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='preview pixels changed'):select_autonomous_construction(text,vision,**args)


def test_failed_actual_image_review_cannot_be_accepted(tmp_path):
    resources=fixture_data(tmp_path/'data')
    result=select_autonomous_construction(TextClient(),VisionClient('failed'),run_id='offline-failed',candidates=[{'candidate_id':'C001'}],observations=[],descriptor_keys=['brightness'],resources=resources,root=tmp_path/'data',receipt_path=tmp_path/'control/receipt.json',task_description='Binary task',max_revisions=0)
    assert result['status']=='deferred'


def test_method_catalog_is_available_but_calibration_requires_authoritative_resources():
    registry=build_default_tool_registry()
    catalog=registry.resolve('inspect_shortcut_methods').execute(EmptyInput(),ToolExecutionContext(run_id='offline',agent_name='identify')).catalog
    assert len(catalog['families'])==15 and len(catalog['papers'])==16
    assert catalog==load_shortcut_knowledge()
    assert registry.is_allowed('quantitative','calibrate_luminance')
    assert not registry.is_allowed('identify','calibrate_luminance')
    tool=registry.resolve('calibrate_luminance')
    with pytest.raises(PermissionError):tool.execute(tool.input_model(mode='normalize_mean',strategy='pooled_mean'),ToolExecutionContext(run_id='offline',agent_name='quantitative'))
    assert 'hidden_truth' not in json.dumps(catalog).lower()
    assert any(f['runtime_status'].startswith('method_memory_only') for f in catalog['families'])


def test_identity_and_path_columns_are_never_scientific_selectors():
    from shortcutminer.experiment.identify_executor import _public_metadata_fields
    request=SimpleNamespace(task=SimpleNamespace(label_column='y'))
    context=SimpleNamespace(metadata_profile={'columns':[{'name':k} for k in ('y','img_id','img_filename','custom_id','custom_path','site')]},visual_discovery=SimpleNamespace(options=SimpleNamespace(sample_id_column='custom_id',sample_path_column='custom_path',split_column='split')))
    assert _public_metadata_fields(request,context)=={'site'}


def test_model_view_requires_frozen_acceptance_and_contains_no_prebuilt_ood(tmp_path):
    from shortcutminer.experiment.autonomous_audit import prepare_autonomous_view,public_rows
    from shortcutminer.experiment.autonomous_construction import _freeze
    data=tmp_path/'discovery';resources=fixture_data(data)
    frame=pd.read_csv(data/'metadata.csv',dtype=str);frame=frame[frame.split=='train'];frame.to_csv(data/'metadata.csv',index=False)
    model=tmp_path/'model';model.mkdir()
    import shutil
    for row in frame.to_dict('records'):shutil.copyfile(data/row['img_filename'],model/row['img_filename'])
    rows=frame.to_dict('records')
    for i,(split,level) in enumerate([('val',30),('val',40),('id',200),('id',220)],start=4):
        Image.new('RGB',(12,12),(level,level,level)).save(model/f'{i}.png')
        rows.append({'img_id':str(i),'img_filename':f'{i}.png','y':str(i%2),'split':split})
    pd.DataFrame(rows).to_csv(model/'metadata.csv',index=False)
    receipt=select_autonomous_construction(TextClient(),VisionClient(),run_id='offline-view',candidates=[{'candidate_id':'C001'}],observations=[],descriptor_keys=['brightness'],resources=resources,root=data,receipt_path=tmp_path/'control/receipt.json',task_description='Binary task',max_revisions=0)
    rejected={**receipt,'status':'deferred'}
    rejected.pop('receipt_hash');rejected=_freeze(tmp_path/'control/rejected.json',rejected)
    with pytest.raises(ValueError,match='precede evaluation'):
        prepare_autonomous_view(rejected,data,model,tmp_path/'never-created')
    assert not (tmp_path/'never-created').exists()
    binding=prepare_autonomous_view(receipt,data,model,tmp_path/'view')
    assert binding['split_counts']=={'train':4,'val':2,'id':2}
    assert binding['proposal_frozen_before_heldout_access'] and not binding['external_ood_used']
    assert set(pd.read_csv(tmp_path/'view/model_metadata.csv').columns)=={'img_id','img_filename','y','split','visual_measurement_c001'}
    assert prepare_autonomous_view(receipt,data,model,tmp_path/'view')==binding
    assert not list((tmp_path/'view').rglob('*ood*'))
    bad=pd.DataFrame(rows);bad.loc[0,'img_filename']='../escape.png';bad.to_csv(model/'metadata.csv',index=False)
    with pytest.raises(ValidationError):public_rows(model)


def test_generic_entry_reaches_standard_five_agent_runner_offline(tmp_path,monkeypatch):
    import importlib.util
    import contextlib
    import sys
    source=Path(__file__).resolve().parents[1]/'infrastructure/experiments/run_autonomous_image_audit.py'
    spec=importlib.util.spec_from_file_location('generic_entry_test',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=tmp_path/'discovery';resources=fixture_data(data)
    frame=pd.read_csv(data/'metadata.csv',dtype=str);frame=frame[frame.split=='train'];frame.to_csv(data/'metadata.csv',index=False)
    model=tmp_path/'model';model.mkdir()
    import shutil
    rows=frame.to_dict('records')
    for row in rows:shutil.copyfile(data/row['img_filename'],model/row['img_filename'])
    for i,(split,level) in enumerate([('val',30),('val',40),('id',200),('id',220)],start=4):
        Image.new('RGB',(12,12),(level,level,level)).save(model/f'{i}.png')
        rows.append({'img_id':str(i),'img_filename':f'{i}.png','y':str(i%2),'split':split})
    pd.DataFrame(rows).to_csv(model/'metadata.csv',index=False)
    profile=Path(__file__).resolve().parents[1]/'configs/experiments/synthetic-corner-patch-shortcutminer-memory-v4.json'
    from shortcutminer.experiment.autonomous_construction import select_autonomous_construction as select
    monkeypatch.setattr(module,'formal_credentials',contextlib.nullcontext)
    monkeypatch.setattr(module,'OpenAICompatibleClient',lambda *a,**k:object())
    monkeypatch.setattr(module,'select_autonomous_construction',lambda c,v,**kw:select(TextClient(),VisionClient(),**kw))
    calls=[]
    def execute(request,*a,**kw):
        calls.append(request)
        directory=tmp_path/'runs'/request.run_id;directory.mkdir(parents=True)
        if request.run_id.endswith('-discovery'):
            (directory/'identify').mkdir()
            (directory/'identify/output.json').write_text(json.dumps({'candidates':[{'candidate_id':'C001','hypothesis':'Observed photometric variation'}]}))
            (directory/'identify/raw_output.json').write_text(json.dumps({'candidates':[]}))
            (directory/'identify/visual_discovery.json').write_text(json.dumps({'sample_observations':[{'descriptors':{'values':{'brightness':.5}}}],'group_summaries':[]}))
        else:
            assert isinstance(a[-1],module.FrozenIdentifyReplay)
        return directory
    monkeypatch.setattr(module,'execute',execute)
    monkeypatch.setattr(sys,'argv',[str(source),'--input',str(data),'--training-input',str(model),'--profile',str(profile),'--run-id','offline-generic','--task','Generic binary task','--runs-root',str(tmp_path/'runs'),'--validation'])
    module.main()
    assert len(calls)==2
    final=calls[-1]
    assert final.metadata['construction_receipt_hash']
    original=json.loads(profile.read_text())
    # Scientific policy and training plan are preserved rather than tuned to planted truth.
    assert final.metadata['promotion_policy']==original['promotion_policy']
    resources=final.metadata['audit_request']['execution_options']
    for key,value in original['execution_options']['adaptive_screening'].items():
        assert resources['adaptive_screening'][key]==value
    for key,value in original['execution_options']['training_options'].items():
        assert resources['training_options'][key]==value
    assert resources['visual_measurement_bindings']=={'C001':'visual_measurement_c001'}
    assert resources['visual_ood_specs']['visual_measurement_c001']['type']=='luminance'


def test_calibration_tool_persists_full_provenance_without_overloading_agent_payload(tmp_path):
    resources=fixture_data(tmp_path/'data')
    tool=build_default_tool_registry().resolve('calibrate_luminance')
    context=ToolExecutionContext(run_id='bounded-calibration',agent_name='quantitative',dataset_root=tmp_path/'data',runs_root=tmp_path/'runs',construction_resources=vars(resources))
    result=tool.execute(tool.input_model(mode='normalize_mean',strategy='pooled_median'),context)
    assert 'image_hashes' not in result.receipt and result.receipt['image_hash_count']==4
    stored=json.loads((tmp_path/'runs/bounded-calibration'/result.receipt['receipt_artifact']).read_text())
    assert len(stored['image_hashes'])==4 and stored['identity']==result.receipt['identity']
    assert '4.png' not in stored['image_hashes']
