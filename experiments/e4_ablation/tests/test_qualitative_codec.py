from copy import deepcopy
import pytest
from experiments.e4_ablation.variants.qualitative_codec import encode_qualitative_descriptors


def fixture():
    samples=[{'sample_id':str(i),'metadata':{'y':i,'visual_measurement_c001':n},
        'byte_size':n,'sample_available':True,'inspection_status':'completed',
        'visual_descriptors':{'intensity':{'mean':n/100}}} for i,n in enumerate([20,30])]
    payload={'request':{'task':{'inputs':{'label_column':'y','metadata_fields':['visual_measurement_c001']}}},
             'evidence_context':{'selected_samples':samples}}
    draft={'sample_ids':['0','1'],'evidence_status':'inconclusive','evidence_summary':'Mixed evidence.',
        'observations':[{'sample_ref':'0','observation':'Original observation.','inspection_quality':'sufficient'}],
        'comparisons':[],'interpretations':[{'interpretation':'No reliance established.','supporting_observation_indexes':[0],
            'supporting_comparison_indexes':[],'confidence':'low'}],
        'counterevidence':[{'sample_refs':['0'],'description':'Retain this contradiction.'}],
        'metadata_evidence':[{'field':'file_size_bytes','sample_refs':['0','1'],'observation':'Size variation; not model reliance.'}],
        'failures':[],'artifact_references':[],'additional_limitations':['No model evidence.'],'recommended_followup':[]}
    return draft,payload


def test_unique_verified_descriptor_alias_preserves_science_and_input():
    draft,payload=fixture();original=deepcopy(draft)
    encoded,changes=encode_qualitative_descriptors(draft,payload)
    assert draft==original
    expected=deepcopy(original);expected['metadata_evidence'][0]['field']='visual_measurement_c001'
    assert encoded==expected and changes[0]['action']=='exact_numeric_alias'


def test_unbound_supplied_descriptor_is_losslessly_encoded_without_changing_indices():
    draft,payload=fixture();draft['metadata_evidence'][0]['field']='intensity_mean'
    encoded,changes=encode_qualitative_descriptors(draft,payload)
    assert encoded['metadata_evidence']==[]
    assert encoded['observations'][0]==draft['observations'][0]
    assert len(encoded['observations'])==3
    assert all(draft['metadata_evidence'][0]['observation'] in o['observation'] for o in encoded['observations'][1:])
    for key in ['evidence_status','evidence_summary','interpretations','counterevidence','additional_limitations']:
        assert encoded[key]==draft[key]
    assert changes[0]['action']=='supplied_descriptor_to_observations'


@pytest.mark.parametrize('problem',['unknown_field','other_task_sample','unavailable_sample','absent_descriptor'])
def test_unverifiable_evidence_remains_for_original_validator_to_reject(problem):
    draft,payload=fixture()
    if problem=='unknown_field':draft['metadata_evidence'][0]['field']='hidden_watermark_state'
    elif problem=='other_task_sample':draft['metadata_evidence'][0]['sample_refs']=['previous-run-id']
    elif problem=='unavailable_sample':payload['evidence_context']['selected_samples'][0]['sample_available']=False
    else:payload['evidence_context']['selected_samples'][0]['byte_size']=None
    encoded,changes=encode_qualitative_descriptors(draft,payload)
    assert encoded==draft and changes==[]


def test_numeric_coincidence_does_not_alias_to_label_or_ambiguous_columns():
    draft,payload=fixture()
    payload['request']['task']['inputs']['metadata_fields'].append('duplicate')
    for s in payload['evidence_context']['selected_samples']:
        s['metadata']['duplicate']=s['byte_size']
    encoded,changes=encode_qualitative_descriptors(draft,payload)
    assert changes[0]['action']=='supplied_descriptor_to_observations'
    assert encoded['metadata_evidence']==[]


def test_actual_structured_boundary_retains_provider_text_and_runs_validator(tmp_path):
    import json
    from experiments.e4_ablation.variants.policies import PolicyMixin
    raw,payload=fixture();captured=[]
    class TransportBase:
        def __init__(self):self._runs_root=tmp_path
        def _invoke_structured(self,**kwargs):
            result=deepcopy(raw);kwargs['validator'](result);return result
    class Client(PolicyMixin,TransportBase):pass
    client=Client(variant='fixed_pipeline',protocol_revision='descriptor-codec-v3')
    def validator(value):
        assert value['metadata_evidence'][0]['field']=='visual_measurement_c001'
        captured.append(deepcopy(value))
    result=client._invoke_structured(agent_name='qualitative',run_id='fixture',payload_for_hash=payload,
        messages=[{'role':'user','content':'evidence'}],system_prompt='contract',validator=validator)
    assert result==captured[0]
    receipt=json.loads(next((tmp_path/'fixture/protocol_encoding').glob('*.json')).read_text())
    assert receipt['provider_output']==raw
    assert receipt['canonical_output']==result and not receipt['validator_bypassed']
    def reject(value):raise ValueError('scientific contract still rejects')
    with pytest.raises(ValueError,match='scientific contract still rejects'):
        client._invoke_structured(agent_name='qualitative',run_id='fixture',payload_for_hash=payload,
            messages=[],system_prompt='',validator=reject)
