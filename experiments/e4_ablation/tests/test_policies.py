import json
from pathlib import Path
from types import SimpleNamespace
import csv
import numpy as np
from PIL import Image
import pytest

from experiments.e4_ablation.variants.policies import (
    routing_draft, restrict_routing_payload, fixed_identify, fixed_measurements,
    ControllerSession, TextClient,
    restrict_request,
    qualitative_contract_guidance, PolicyMixin,
)


def test_qualitative_guidance_uses_only_authorized_current_metadata():
    p={'request':{'task':{'inputs':{'label_column':'y','metadata_fields':['source']}}},
       'candidate':{'name':'invented_color_field'}}
    text=qualitative_contract_guidance('qualitative',p)
    assert '["source", "y"]' in text
    assert 'invented_color_field' not in text
    assert 'metadata_evidence=[]' in text
    assert qualitative_contract_guidance('identify',p)==''
    assert qualitative_contract_guidance('qualitative',{'mode':'construction_semantic_review'})==''


def test_protocol_guidance_applies_at_multimodal_boundary_without_mutating_request():
    class Capture:
        def __init__(self):pass
        def _invoke_structured(self,**kwargs):return kwargs
    class Client(PolicyMixin,Capture):pass
    payload={'request':{'task':{'inputs':{'label_column':'y','metadata_fields':[]}}}}
    messages=[{'role':'user','content':'current images'}]
    client=Client(variant='fixed_pipeline',protocol_revision='metadata-contract-v2')
    received=client._invoke_structured(agent_name='qualitative',payload_for_hash=payload,messages=messages,system_prompt='original')
    assert received['payload_for_hash'] is payload
    assert messages==[{'role':'user','content':'current images'}]
    assert '["y"]' in received['messages'][0]['content']
    assert 'original' in received['system_prompt']
    legacy=Client(variant='fixed_pipeline')._invoke_structured(agent_name='qualitative',payload_for_hash=payload,messages=messages,system_prompt='original')
    assert legacy['messages'] is messages and legacy['system_prompt']=='original'
from shortcutminer.experiment.autonomous_construction import (
    select_autonomous_construction,preview_construction,
)
from shortcutminer.intervention.protocol import LuminanceTransform
from experiments.e4_ablation.scripts.score_adjudication import score
from experiments.e4_ablation.scripts.collect_results import collect_arm


def routing_payload():
    return {'request':{'routing_context':{'audit_options':{'allowed_targets':['quantitative','qualitative']}}},
        'executable_candidate_contexts':[{'candidate_id':'C001','allowed_quantitative_checks':['association_test','task_model_training','task_model_evaluation','ood_evaluation'],
            'allowed_qualitative_checks':['sample_inspection','counterexample_inspection']} ]}


def test_fixed_routing_keeps_executable_training_and_ood():
    value=routing_draft(routing_payload(),'fixed_pipeline')['decisions'][0]
    assert value['route']=='both'
    assert value['quantitative']['requested_checks']==['association_test','task_model_training','task_model_evaluation','ood_evaluation']


def test_counter_ablation_removes_search_without_removing_observed_evidence():
    original=routing_payload();original['observed_counterevidence']=[{'direction':'contradicts','sample':'s1'}]
    filtered=restrict_routing_payload(original,'without_counterevidence')
    assert filtered['observed_counterevidence']==original['observed_counterevidence']
    assert 'counterexample_inspection' not in filtered['executable_candidate_contexts'][0]['allowed_qualitative_checks']
    assert 'counterexample_inspection' in original['executable_candidate_contexts'][0]['allowed_qualitative_checks']


@pytest.mark.parametrize('target',['quantitative','qualitative'])
def test_disabled_branch_not_synthesized(target):
    payload=restrict_routing_payload(routing_payload(),'without_'+target)
    payload['request']['routing_context']['audit_options']['allowed_targets']=[t for t in ('quantitative','qualitative') if t!=target]
    d=routing_draft(payload,'without_'+target)['decisions'][0]
    assert d[target] is None


def test_request_switches_use_valid_disabled_learning_policy():
    from shortcutminer.experiment.learned_shortcuts import LearningPolicy
    class Request:
        def model_dump(self,**kwargs):return {'metadata':{},'executive_config':{}}
        @classmethod
        def model_validate(cls,value):return value
    value=restrict_request(Request(),'without_quantitative')
    assert not LearningPolicy.model_validate(value['metadata']['knowledge_policy']).enabled
    assert value['executive_config']['audit_options']['allowed_targets']==['qualitative']


def test_fixed_discovery_uses_current_descriptors_without_truth_or_candidates():
    p={'request':{'budget':{'max_candidates':5}},'discovery_context':{'visual_discovery':{'descriptor_differences':[
        {'descriptor':key,'group_a':'y=0','group_b':'y=1','standardized_descriptive_difference':.5}
        for key in ('border_mean_intensity','border_rgb_mean_blue','edge_density','intensity_mean')]}}}
    result=fixed_identify(p)
    assert [c['cue']['name'] for c in result['candidates']]==['border_mean_intensity','edge_density','intensity_mean']
    candidates=[{**c,'candidate_id':f'C00{i+1}'} for i,c in enumerate(result['candidates'])]
    plan=fixed_measurements({'frozen_candidates':candidates,'available_descriptor_keys':['border_mean_intensity']})
    assert plan['measurements'][0]['status']=='measurable'
    assert plan['measurements'][1]['status']=='deferred'
    assert plan['measurements'][1]['kind'] is None


def test_single_controller_memory_is_shared_and_persistent(tmp_path):
    session=ControllerSession(tmp_path/'memory.json')
    session.record('identify',{'request':'opaque'},{'candidate':'cue'})
    session.record('qualitative',{'request':'opaque'},{'direction':'contradicts'})
    resumed=ControllerSession(tmp_path/'memory.json')
    assert len(resumed.context())==2
    assert resumed.context()[1]['response']['direction']=='contradicts'


def test_live_collector_ignores_runtime_snapshots_and_telemetry_aggregate(tmp_path):
    folder=tmp_path/'run/model_calls';folder.mkdir(parents=True)
    (folder/'runtime_snapshot.json').write_text(json.dumps({'model':'configured'}))
    (folder/'telemetry.json').write_text(json.dumps({'invocations':0}))
    result=collect_arm({'directory':str(tmp_path),'case':'fixture','variant':'full','run_id':'run'})
    assert result['text_provider_calls']==0 and result['status']=='PENDING'


def test_unknown_truth_is_not_silently_scored_as_false():
    row={'case':'opaque','variant':'full','candidates':[{'candidate_id':'C001'},{'candidate_id':'C002'}],
        'assessments':{'C001':'supported','C002':'partially_supported'}}
    j={'positive_mechanism_ids':['m1'],'candidates':{'C001':{'verdict':'true',
        'mechanism_id':'m1','discovery_match':True,'independent_validation_reference':'external/frozen-check.json'}}}
    result=score(row,j)
    assert result['discovery_recall']==1
    assert result['verified_discovery_precision'] is None
    assert result['unknown_candidates']==['C002']
    assert result['precision_lower_bound']==.5 and result['precision_upper_bound']==1


def test_partial_support_does_not_become_negative_control_false_confirmation():
    row={'case':'opaque','variant':'full','candidates':[{'candidate_id':'C001'}],
        'assessments':{'C001':'partially_supported'}}
    j={'negative_mechanism_ids':['n1'],'candidates':{'C001':{'verdict':'false','mechanism_id':'n1'}}}
    assert score(row,j)['false_positive_rate_on_declared_negative_mechanisms']==0


def make_images(tmp_path):
    rows=[]
    for i in range(20):
        ref=f'{i:02}.png';Image.fromarray(np.full((8,8,3),30+i*7,np.uint8)).save(tmp_path/ref)
        rows.append({'path':ref,'split':'train'})
    rows.append({'path':'heldout-must-not-open.png','split':'id'})
    with (tmp_path/'metadata.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','split']);w.writeheader();w.writerows(rows)
    return SimpleNamespace(model_metadata_file='metadata.csv',split_column='split',train_split_values=['train'],sample_path_column='path',max_train_samples=20)


def test_without_counter_preview_is_random_and_never_opens_heldout(tmp_path):
    resources=make_images(tmp_path)
    op=LuminanceTransform(type='luminance',mode='offset',offset=10)
    rows,_=preview_construction(op,resources,tmp_path,tmp_path/'preview',quality_policy='training-construction-review-v2',seek_counterexamples=False)
    assert len(rows)==12 and len({r['source_reference'] for r in rows})==12
    assert all(r['preview_selection']=='seeded_random' for r in rows)
    assert [r['source_reference'] for r in rows]!=[f'{i:02}.png' for i in range(12)]


def test_without_qualitative_constructor_never_calls_visual_model(tmp_path):
    resources=make_images(tmp_path)
    class Client:
        def invoke_agent_validated(self,name,payload,*,validator,**kwargs):
            if name=='quantitative':
                v=dict(status='selected',candidate_id='C001',descriptor_key='intensity_mean',
                    transformation={'type':'luminance','mode':'offset','offset':10},luminance_calibration=None,
                    rationale='Scoped test.',intended_invariants=['Geometry unchanged.'],limitations=['Semantics unverified.'])
            else:
                assert payload['visual_review'] is None
                v=dict(decision='accept',rationale='Sensitivity diagnostic only.',limitations=['No semantic review.'])
            validator(v);return v
    class ForbiddenVision:
        def invoke_multimodal(self,**kwargs):raise AssertionError('Visual review must be disabled')
    result=select_autonomous_construction(Client(),ForbiddenVision(),run_id='fixture',candidates=[{'candidate_id':'C001'}],
        observations=[],descriptor_keys=['intensity_mean'],resources=resources,root=tmp_path,
        receipt_path=tmp_path/'receipt/construction.json',task_description='Opaque classification.',disable_qualitative=True)
    assert result['status']=='selected' and result['visual_review'] is None
    assert result['qualitative_review_status']=='not_performed_ablation'
    assert result['semantic_preservation']=='unverified'
