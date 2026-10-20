from experiments.eswa_20261019.resource_policy import bound_visual_tasks,prevalidation_memory_for_constructions
from experiments.eswa_20261019.resource_policy import single_probe_profile,completed_discovery
import json
import pytest
from types import SimpleNamespace
import hashlib
from experiments.eswa_20261019.resource_policy import publish_probe_cache


def test_cache_identity_survives_json_roundtrip(tmp_path):
    from experiments.eswa_20261019.run_one import cache_identity
    path=tmp_path/'image';path.write_bytes(b'image')
    key,content=cache_identity([path],['0'],[path],['0'],{'seeds':[42]},{'resnet18':'hash'})
    assert json.loads(json.dumps(content))==content
    assert len(key)==64


def test_study_cache_materializes_backend_references_and_rejects_corruption(tmp_path):
    cache=tmp_path/'cache';source=cache/'resnet18-seed42';source.mkdir(parents=True)
    (source/'best.npz').write_bytes(b'known-weights')
    digest=hashlib.sha256(b'known-weights').hexdigest()
    (source/'profile.json').write_text(json.dumps({'weights_sha256':digest,'classes':['0','1']}))
    model=SimpleNamespace(profiles=[{'architecture':'resnet18','seed':42,'weights_sha256':digest}])
    destination=tmp_path/'backend'
    publish_probe_cache(model,cache,destination)
    assert (destination/'resnet18-seed42/best.npz').read_bytes()==b'known-weights'
    assert json.loads((destination/'resnet18-seed42/profile.json').read_text())['weights_sha256']==digest
    (source/'best.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='weights changed'):publish_probe_cache(model,cache,destination)


def test_single_probe_preserves_per_candidate_alpha_and_other_resources():
    profile={'promotion_policy':{'max_probe_candidates':3},'execution_options':{
        'adaptive_screening':{'family_alpha':0.05,'candidate_budget':3,'minimum_effect':0.02},
        'training_options':{'seeds':[42]}}}
    new=single_probe_profile(profile)
    assert new['promotion_policy']['max_probe_candidates']==new['execution_options']['adaptive_screening']['candidate_budget']==1
    assert new['execution_options']['adaptive_screening']['family_alpha']==0.05/3
    assert new['execution_options']['training_options']==profile['execution_options']['training_options']
    assert profile['execution_options']['adaptive_screening']['candidate_budget']==3


def test_discovery_chain_rejects_cycles_and_external_paths(tmp_path):
    first=tmp_path/'first';first.mkdir()
    (first/'CONTINUATION.json').write_text(json.dumps({'previous_attempt':str(first)}))
    with pytest.raises(ValueError,match='Cyclic'):completed_discovery(first,'discovery',tmp_path)
    (first/'CONTINUATION.json').write_text(json.dumps({'previous_attempt':str(tmp_path.parent)}))
    with pytest.raises(ValueError):completed_discovery(first,'discovery',tmp_path)


def test_visual_budget_preserves_numeric_capabilities_and_prefers_approved_construction():
    contexts=[{'candidate_id':str(i),'allowed_qualitative_checks':['sample_inspection'],
        'allowed_quantitative_checks':['association_test'],'structured_intervention_spec_available':i==2} for i in range(4)]
    payload={'executable_candidate_contexts':contexts};new=bound_visual_tasks(payload,1)
    assert [c['candidate_id'] for c in new['executable_candidate_contexts'] if c['allowed_qualitative_checks']]==['2']
    assert all(c['allowed_quantitative_checks']==['association_test'] for c in new['executable_candidate_contexts'])
    assert all(c['allowed_qualitative_checks'] for c in contexts)


def test_development_memory_fork_omits_unbudgeted_constructions_and_failed_validation():
    turns=[{'phase_adapter':'identify','response':{'candidates':[]}},
        {'phase_adapter':'executive','response':{'decisions':[]}},
        {'phase_adapter':'quantitative','response':{'candidate_id':'C1','descriptor_key':'edge'}},
        {'phase_adapter':'qualitative','response':{'status':'reviewed'}},
        {'phase_adapter':'quantitative','response':{'candidate_id':'C2','descriptor_key':'color'}},
        {'phase_adapter':'qualitative','response':{'status':'reviewed2'}},
        {'phase_adapter':'executive','response':{'decisions':[]}},
        {'phase_adapter':'quantitative','response':{'status':'measurement'}}]
    assert prevalidation_memory_for_constructions(turns,{'C1'})==turns[:4]
    assert len(turns)==8
