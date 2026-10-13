"""Real low-cost execution and opt-in/legacy compatibility; no provider/GPU calls."""
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from shortcutminer.experiment.candidate_validation import (
    CandidateMeasurement, freeze_candidate_measurements, prepare_candidate_complete_view,
    evaluate_measurement, label_group_preparation, write_candidate_dispositions,
)
from shortcutminer.experiment.autonomous_audit import prepare_autonomous_view, sha
from shortcutminer.experiment.autonomous_construction import _hash
from shortcutminer.runtime.evidence_context import evidence_context, validate_approved_probe_routing


def dataset(tmp_path):
    discovery=tmp_path/'discovery'; model=tmp_path/'model'
    discovery.mkdir();model.mkdir();rows=[]
    for i in range(60):
        split='train' if i<40 else 'val' if i<50 else 'id'; y=i%10
        pixels=np.full((16,16,3),20+y*10,np.uint8)
        rng=np.random.default_rng(i)
        pixels[:1+i%12,:]=rng.integers(0,256,(1+i%12,16,3),dtype=np.uint8)
        row={'img_id':str(i),'img_filename':f'images/{i}.png','y':y,'split':split}
        f=model/row['img_filename'];f.parent.mkdir(exist_ok=True);Image.fromarray(pixels).save(f)
        if split=='train':
            f=discovery/row['img_filename'];f.parent.mkdir(exist_ok=True);Image.fromarray(pixels).save(f)
        rows.append(row)
    for root,subset in [(model,rows),(discovery,rows[:40])]:
        with (root/'metadata.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(subset)
    return discovery,model


def proposal():
    specs=[]
    for cid,key in [('C001','edge_density'),('C002','luminance_y_mean'),('C003','center_mean_intensity'),('C004','file_size_bytes')]:
        specs.append({'candidate_id':cid,'status':'measurable','kind':'descriptor','descriptor_key':key,
            'rationale':'Public training observations support this proxy.','limitations':['Proxy may reflect task semantics.']})
    specs.append({'candidate_id':'C005','status':'not_applicable','rationale':'Training split is constant; no acquisition metadata.','limitations':['Not a negative shortcut verdict.']})
    return {'policy_version':'candidate-complete-v1','measurements':specs}


class PlanClient:
    def __init__(self, raw=None):self.calls=0;self.raw=raw or proposal()
    def invoke_agent_validated(self,name,payload,*,validator,**kwargs):
        self.calls+=1;assert name=='quantitative';assert 'y' not in payload['available_descriptor_keys']
        validator(self.raw);return deepcopy(self.raw)


def freeze(tmp_path,client=None):
    discovery,model=dataset(tmp_path)
    plan,receipt=freeze_candidate_measurements(client or PlanClient(),root=discovery,
        candidates=[{'candidate_id':f'C00{i}'} for i in range(1,6)],observations=[],
        receipt_path=tmp_path/'plan.json',prompt='Training-only fixture.')
    return discovery,model,plan,receipt


def construction(discovery,receipt,candidate='C002'):
    spec=next(s for s in receipt['plan']['measurements'] if s['candidate_id']==candidate)
    value={'status':'selected','proposal':{'candidate_id':candidate,'descriptor_key':spec['descriptor_key']},
        'transformation':{'type':'luminance','mode':'normalize_mean','target_mean':70},
        'truth_seen':False,'heldout_seen':False,'construction_metadata_sha256':sha(discovery/'metadata.csv'),
        'training_source_hashes':receipt['training_source_hashes']}
    value['receipt_hash']=_hash(value);return value


def test_all_candidates_get_bindings_even_without_ood(tmp_path):
    d,m,plan,receipt=freeze(tmp_path)
    result=prepare_candidate_complete_view(receipt,{},discovery_root=d,training_root=m,view=tmp_path/'view')
    assert set(result['visual_measurement_bindings'])=={'C001','C002','C003','C004'}
    assert result['structured_specs']=={} and result['candidate_resolutions']['C005']['status']=='not_applicable'
    assert set(receipt['training_source_hashes'])=={f'images/{i}.png' for i in range(40)}
    assert receipt['heldout_seen'] is False


def test_actual_five_agent_graph_executes_all_four_cheap_checks_without_training(tmp_path):
    from shortcutminer.audit import build_audit_run_request
    from shortcutminer.protocol.audit import AuditRequest
    from shortcutminer.runner import run,build_local_tool_runtimes
    from shortcutminer.runtime.demo import DemoAgentClient
    d,m,plan,receipt=freeze(tmp_path);v=tmp_path/'view'
    binding=prepare_candidate_complete_view(receipt,{},discovery_root=d,training_root=m,view=v)
    class Client(DemoAgentClient):
        def _identify(self,payload):
            return {'candidates':[{'cue':{'family':'appearance','name':f'visual{i}'},'description':'Public appearance proxy.',
                'observation':'Training image measurements vary.','hypothesis':'Public cue may correlate with labels.',
                'evidence_sources':['image_statistics'],'priority':'high','additional_limitations':[]} for i in range(1,5)],'additional_limitations':[]}
        def _executive(self,payload):
            out=super()._executive(payload)
            for decision in out['decisions']:
                decision['quantitative']['requested_checks']=['association_test']
                decision['qualitative']['requested_checks']=['sample_inspection']
            return out
        def _qualitative(self,payload):
            out=super()._qualitative(payload)
            out['evidence_status']='inconclusive'
            out['metadata_evidence'][0]['field']='y'
            out['metadata_evidence'][0]['observation']='Ten label groups are explicitly supplied.'
            return out
        def _summarize(self,payload):
            return {'overview':'All four candidates received cheap checks; no model reliance claim.',
                'findings':[{'candidate_ref':f'C00{i}','assessment':'inconclusive','summary':'Cheap evidence only.',
                    'additional_limitations':[],'recommendations':[]} for i in range(1,5)],'additional_limitations':[]}
    options=dict(sample_id_column='img_id',sample_path_column='img_filename',split_column='split',
        model_metadata_file='model_metadata.csv',train_split_values=['train'],validation_split_values=['val'],
        test_split_values=['id'],evaluation_split_values=['id'],model_backend='reference_image_linear',
        feature_version='reference_image_features_v1',max_train_samples=40,max_samples=10,
        association_variable_mode='numeric_categorical',visual_measurement_bindings=binding['visual_measurement_bindings'])
    request=build_audit_run_request(AuditRequest(dataset_path=str(v),metadata_file='discovery_metadata.csv',
        label_information={'source':'metadata_column','column':'y'},execution_options=options),run_id='complete-fixture',runs_root=tmp_path/'runs')
    request=request.model_copy(update={'qualitative_config':label_group_preparation('y'),'metadata':{**request.metadata,
        'promotion_policy':{'policy_version':'budgeted-test','calibration_status':'test_only','strategy':'budgeted_paired_probe','max_probe_candidates':3}}})
    tools=build_local_tool_runtimes(run_id=request.run_id,dataset_root=v,runs_root=request.runs_root)
    assert run(request,agent_client=Client(),tool_runtimes=tools).status=='COMPLETED'
    runroot=Path(request.runs_root)/request.run_id
    for cid in ['C001','C002','C003','C004']:
        result=json.loads((runroot/f'quantitative/TASK-{cid}-QT/result.json').read_text(encoding='utf-8'))['payload']
        assert any(z['experiment_type']=='statistical_check' and z['execution_outcome']=='completed' for z in result['measurements'])
        context=json.loads((runroot/f'qualitative/TASK-{cid}-QL/input.json').read_text(encoding='utf-8'))['evidence_context']
        assert len(context['group_summaries'])==10
        assert all(z['group']!='ungrouped' for z in context['selected_samples'])
    assert not list(runroot.glob('models/task/*/model.json'))


def test_legacy_view_remains_single_binding_and_complete_resume_detects_tampering(tmp_path):
    d,m,plan,receipt=freeze(tmp_path);accepted=construction(d,receipt)
    old=prepare_autonomous_view(accepted,d,m,tmp_path/'legacy')
    with (tmp_path/'legacy/discovery_metadata.csv').open() as f:assert csv.DictReader(f).fieldnames==['img_id','img_filename','y','split','visual_measurement_c002']
    new=prepare_candidate_complete_view(receipt,{'C002':accepted},discovery_root=d,training_root=m,view=tmp_path/'new')
    assert len(new['visual_measurement_bindings'])==4 and len(new['structured_specs'])==1
    assert prepare_candidate_complete_view(receipt,{'C002':accepted},discovery_root=d,training_root=m,view=tmp_path/'new')==new
    (tmp_path/'new/images/0.png').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash changed'):prepare_candidate_complete_view(receipt,{'C002':accepted},discovery_root=d,training_root=m,view=tmp_path/'new')


@pytest.mark.parametrize('bad',['label','missing_candidate','duplicate_candidate'])
def test_no_label_binding_or_incomplete_agent_plan(tmp_path,bad):
    raw=proposal()
    if bad=='label':raw['measurements'][0]['descriptor_key']='y'
    elif bad=='missing_candidate':raw['measurements'].pop()
    else:raw['measurements'].append(raw['measurements'][0])
    with pytest.raises(ValueError):freeze(tmp_path,PlanClient(raw))


def test_ratio_missing_denominator_never_becomes_zero():
    spec=CandidateMeasurement(candidate_id='C001',status='measurable',kind='ratio',descriptor_key='a',denominator_key='b',rationale='Ratio.',limitations=['Proxy.'])
    assert evaluate_measurement(spec,{'a':2.,'b':0.}) is None
    assert evaluate_measurement(spec,{'a':2.,'b':4.})==.5


def test_constant_descriptor_closed_with_reason_and_plan_resume_frozen(tmp_path):
    raw=proposal();raw['measurements'][0]['descriptor_key']='width';client=PlanClient(raw)
    d,m,plan,receipt=freeze(tmp_path,client)
    assert receipt['resolutions']['C001']['status']=='constant_training_measurement'
    freeze_candidate_measurements(client,root=d,candidates=[{'candidate_id':f'C00{i}'} for i in range(1,6)],
        observations=[],receipt_path=tmp_path/'plan.json',prompt='Training-only fixture.')
    assert client.calls==1
    Image.new('RGB',(16,16)).save(d/'images/0.png')
    with pytest.raises(ValueError,match='changed'):freeze_candidate_measurements(client,root=d,
        candidates=[{'candidate_id':f'C00{i}'} for i in range(1,6)],observations=[],receipt_path=tmp_path/'plan.json',prompt='Training-only fixture.')


def test_context_delivery_preserves_metrics_and_canonical_evidence():
    payload={'metrics':{'sample_count':1024,'accuracy_drop':.002},'uncertainty':{'lower':-.01,'upper':.014},
        'looks':[{'sample_count':1024,'decision':'stop_low_effect','sample_ids':['x'*1000]*1024}],
        'pixel_evidence':[{'source':'x'*1000}]*1024}
    original=deepcopy(payload);compact,record=evidence_context(payload)
    assert payload==original and compact['metrics']==payload['metrics'] and compact['uncertainty']==payload['uncertainty']
    assert compact['looks'][0]['decision']=='stop_low_effect' and record['compact_characters']<record['original_characters']


def test_disposition_does_not_invent_a_negative_verdict(tmp_path):
    d,m,plan,receipt=freeze(tmp_path)
    results=write_candidate_dispositions(tmp_path/'run',receipt,{})
    assert all(not r['absence_of_shortcut_proven'] for r in results)
    assert results[-1]['cheap_state']=='no_executable_measurement'


def test_only_approved_executable_ood_requires_model_plan():
    from shortcutminer.tooling.agent_loop import ProviderContractValidationError
    checks=['association_test','task_model_training','task_model_evaluation','ood_evaluation']
    payload={'executable_candidate_contexts':[
        {'candidate_id':'C001','structured_intervention_spec_available':False,'allowed_quantitative_checks':checks},
        {'candidate_id':'C002','structured_intervention_spec_available':True,'allowed_quantitative_checks':checks}]}
    draft={'decisions':[
        {'candidate_id':'C001','quantitative':{'requested_checks':['association_test']}},
        {'candidate_id':'C002','quantitative':{'requested_checks':['association_test']}}]}
    with pytest.raises(ProviderContractValidationError):validate_approved_probe_routing(payload,draft)
    draft['decisions'][1]['quantitative']['requested_checks']=checks
    validate_approved_probe_routing(payload,draft)
