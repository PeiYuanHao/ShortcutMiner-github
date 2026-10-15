import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image

from shortcutminer.experiment.representative_discovery import inspect_representative
from shortcutminer.inspection.semantic import MockSemanticVisualInspector
from shortcutminer.protocol.visual_discovery import SemanticObservation,VisualDiscoveryOptions
from shortcutminer.intervention.protocol import BoundedChannelRatioTransform
from shortcutminer.intervention.bounded_color import transform
from shortcutminer.experiment.construction_quality import preview_sources
from shortcutminer.experiment.autonomous_construction import select_autonomous_construction


def test_representative_nested_coverage_stops_and_resumes_without_calls(tmp_path):
    records=[];observations={}
    for g in range(3):
        for i in range(12):
            sid=f'g{g}-{i}'; ref=sid+'.png'
            Image.fromarray(np.full((8,8,3),80+i,np.uint8)).save(tmp_path/ref)
            records.append(SimpleNamespace(decoded=True,group=str(g),sample_id=sid,sample_reference=ref))
            observations[sid]=[SemanticObservation(sample_id=sid,sample_reference=ref,
                observation_type='foreground_object',description='Visible subject.',confidence='medium',limitations=[])]
    class CoverageClient:
        calls=0
        def invoke_agent_validated(self,name,payload,*,validator,**kwargs):
            self.calls+=1
            value={'decision':'expand' if self.calls==1 else 'stop','rationale':'Training coverage review.',
                'unresolved_observation_questions':[],'limitations':['No model reliance claim.']}
            assert all('g' in k for k in payload['group_by_sample'])
            validator(value);return value
    inspector=MockSemanticVisualInspector(observations)
    inspector.coverage_client=CoverageClient();inspector.coverage_runs_root=tmp_path/'runs'
    request=SimpleNamespace(run_id='fixture',dataset=SimpleNamespace(root=str(tmp_path),name='opaque'),
        task=SimpleNamespace(model_dump=lambda **kwargs:{'type':'classification'}))
    result=inspect_representative(inspector,request,records)
    assert len(result)==24 and inspector.coverage_client.calls==2
    receipt=json.loads((tmp_path/'runs/fixture/identify/representative-coverage/receipt.json').read_text())
    assert receipt['semantic_sample_count']==24 and not receipt['budget_exhausted']
    assert all(n==8 for n in receipt['looks'][-1]['coverage_per_class'].values())
    before=len(inspector.calls)
    assert inspect_representative(inspector,request,records)==result
    assert len(inspector.calls)==before and inspector.coverage_client.calls==2
    # Replacing an observed source must invalidate a cached decision.
    Image.fromarray(np.full((8,8,3),255,np.uint8)).save(tmp_path/inspector.calls[0].samples[0].sample_reference)
    with pytest.raises(ValueError,match='input changed'):
        inspect_representative(inspector,request,records)


def test_bounded_color_preserves_luminance_and_changes_ratio_without_geometry_loss():
    rng=np.random.default_rng(2026)
    a=rng.integers(0,256,(30,40,3),dtype=np.uint8)
    a[:,:,2]//=3
    spec=BoundedChannelRatioTransform(type='bounded_channel_ratio',mode='normalize_mean',
        numerator_channel='red',denominator_channel='blue',center_ratio=1,max_log_shift=.4)
    b=transform(a,spec); w=np.array([.299,.587,.114])
    assert a.shape==b.shape and b.dtype==np.uint8
    assert np.max(np.abs(a@w-b@w))<=.500001
    assert abs(b[:,:,0].mean()/b[:,:,2].mean()-1)<abs(a[:,:,0].mean()/a[:,:,2].mean()-1)
    assert not np.array_equal(a,b)


def test_preview_covers_random_and_both_training_extremes_only(tmp_path):
    import csv
    rows=[]
    for i in range(24):
        Image.fromarray(np.full((8,8,3),[20+i*8,80,60],np.uint8)).save(tmp_path/f'{i:02}.png')
        rows.append({'path':f'{i:02}.png','split':'train'})
    rows.append({'path':'must-not-open.png','split':'id'})
    with (tmp_path/'metadata.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','split']);w.writeheader();w.writerows(rows)
    resources=SimpleNamespace(model_metadata_file='metadata.csv',sample_path_column='path',split_column='split',
        train_split_values=['train'],max_train_samples=24)
    operation=BoundedChannelRatioTransform(type='bounded_channel_ratio',mode='normalize_mean',
        numerator_channel='red',denominator_channel='blue',center_ratio=1,max_log_shift=.3)
    chosen=preview_sources(operation,resources,tmp_path)
    assert len(chosen)==12 and len(set(r for r,p in chosen))==12
    assert {'00.png','23.png'}.issubset({r for r,p in chosen})
    assert preview_sources(operation,resources,tmp_path)==chosen


def test_reviewed_construction_rejects_damaged_pairs_before_executive(tmp_path):
    from test_intervention_alignment import inputs,spec,Client
    resources=inputs(tmp_path)
    class ReviewedClient(Client):
        def invoke_agent_validated(self,name,payload,*,validator,**kwargs):
            assert name=='quantitative', 'Damaged construction must never reach Executive acceptance'
            v={'status':'selected','candidate_id':'C003','descriptor_key':'ratio:rgb_mean_red/rgb_mean_blue',
                'channel_ratio_calibration':{'mode':'normalize_mean','strategy':'pooled_median','max_log_shift':.3},
                'mechanism_scope':'global_color_cast','mechanism_test':'Change a scoped channel-ratio cue.',
                'remaining_confounds':['Natural subject colors.'],'training_selection_criteria':'Training appearance only.',
                'rationale':'Bound chromatic shift.','intended_invariants':['Luminance'],'limitations':['No semantic certificate.']}
            validator(v);return v
    class DamagedVision:
        def invoke_multimodal(self,*,payload_for_hash,validator,**kwargs):
            v={'status':'failed','reviewed_pair_ids':[r['pair_id'] for r in payload_for_hash['training_pairs']],
                'cue_change_observation':'Changed.','task_content_observation':'Details lost.',
                'artifacts_observation':'Severe artifacts.','limitations':['Fixture only.'],
                'mechanism_specificity':'partial','task_preservation':'damaged','recommended_action':'revise',
                'preservation_risks':['Task evidence loss.']}
            validator(v);return v
    result=select_autonomous_construction(ReviewedClient(),DamagedVision(),run_id='fixture',
        candidates=[{'candidate_id':'C003'}],observations=[],descriptor_keys=['ratio:rgb_mean_red/rgb_mean_blue'],
        resources=resources,root=tmp_path,receipt_path=tmp_path/'receipt/construction.json',task_description='Opaque classification.',
        measurement_binding=spec(),quality_policy='training-construction-review-v2',max_revisions=0)
    assert result['status']=='deferred' and 'quality_review_requires_revision' in result['feedback'][0]


def test_legacy_discovery_defaults_unchanged():
    options=VisualDiscoveryOptions()
    assert options.max_samples==40 and options.sampling_strategy=='auto'
    assert 'quality_policy' not in options.model_dump()


def test_representative_profiler_scans_all_numeric_rows_but_bounds_semantic_images(tmp_path):
    import csv
    from test_visual_discovery import _profile
    from shortcutminer.experiment.identify_context import build_identify_context_from_profile
    from shortcutminer.experiment.identify_executor import build_identify_agent_payload
    from shortcutminer.experiment.dataset_profiler import build_dataset_profile
    from test_visual_discovery import _request
    rows=[];observations={}
    for i in range(120):
        sid=f's{i:03}';ref=sid+'.png'
        Image.fromarray(np.full((8,8,3),i+20,np.uint8)).save(tmp_path/ref)
        rows.append({'sample_id':sid,'sample_path':ref,'label':i%3})
        observations[sid]=[SemanticObservation(sample_id=sid,sample_reference=ref,observation_type='foreground_object',
            description='Visible object.',confidence='medium',limitations=[])]
    with (tmp_path/'metadata.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    class StopClient:
        def invoke_agent_validated(self,name,payload,*,validator,**kwargs):
            value={'decision':'stop','rationale':'Sufficient scoped observations.','limitations':['Descriptive only.']}
            validator(value);return value
    inspector=MockSemanticVisualInspector(observations);inspector.coverage_client=StopClient()
    visual=_profile(tmp_path,max_samples=120,options=VisualDiscoveryOptions(max_samples=120,
        sampling_strategy='class_representative_v1',sample_id_column='sample_id',sample_path_column='sample_path',semantic_enabled=True),semantic_inspector=inspector)
    assert visual.coverage.decoded_samples==120 and visual.semantic.available
    assert len(visual.semantic.observations)==12
    assert all(g.decoded_count==40 for g in visual.group_summaries)
    req=_request(tmp_path,max_samples=120);profile=build_dataset_profile(req,seed=17)
    context=build_identify_context_from_profile(req,seed=17,profile=profile,visual_discovery=visual)
    payload=build_identify_agent_payload(req,context)
    delivered=payload['discovery_context']['visual_discovery']
    assert len(delivered['sample_observations'])==12
    assert delivered['numeric_context_delivery']['numeric_decoded_count']==120
    assert len(context.visual_discovery.sample_observations)==120


def test_bounded_operation_is_accepted_by_audit_resources_and_engine(tmp_path):
    from pydantic import TypeAdapter
    from shortcutminer.intervention.protocol import ImageInterventionSpec
    from shortcutminer.protocol.audit import AuditExecutionOptions
    from shortcutminer.intervention.engine import InterventionEngine
    operation={'type':'bounded_channel_ratio','mode':'normalize_mean','numerator_channel':'red',
        'denominator_channel':'blue','center_ratio':1,'max_log_shift':.3}
    parsed=TypeAdapter(ImageInterventionSpec).validate_python(operation)
    options=TypeAdapter(AuditExecutionOptions.model_fields['visual_ood_specs'].annotation).validate_python({'cue':operation})
    assert options['cue'].type=='bounded_channel_ratio'
    Image.fromarray(np.full((8,8,3),[150,80,30],np.uint8)).save(tmp_path/'source.png')
    quality=InterventionEngine().apply(source_path=tmp_path/'source.png',output_path=tmp_path/'changed.png',
        transformation=parsed,dataset_root=tmp_path)
    assert quality.same_dimensions and quality.pixel_change_fraction>0
