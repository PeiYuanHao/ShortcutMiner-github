import itertools
import numpy as np
import pytest
from pydantic import ValidationError
from shortcutminer.experiment.paired_confidence_sequence import log_e_value, paired_betting_interval
from shortcutminer.experiment.adaptive_screening import run_probe
from shortcutminer.protocol.model_reliance import AdaptivePairedScreeningOptions


def options(**updates):
    values = dict(minimum_effect=.02, family_alpha=.05, candidate_budget=3,
                  pair_checkpoints=[128,256,512,1024,1500], probe_fraction=.75,
                  sensitivity_effects=[.01,.02,.05], confidence_method='paired-betting-cs-v1')
    return AdaptivePairedScreeningOptions(**(values | updates))


def test_exact_martingale_expectation_and_optional_stopping_small_tree():
    probabilities = [.1, .6, .3]
    mu = .2
    expectation = 0.
    crossed = 0.
    for sequence in itertools.product(range(3), repeat=7):
        probability = float(np.prod([probabilities[x] for x in sequence]))
        counts = np.zeros(3, dtype=int)
        hit = False
        for x in sequence:
            counts[x] += 1
            hit |= log_e_value(mu, counts) >= np.log(1/.05)
        expectation += probability * np.exp(log_e_value(mu, counts))
        crossed += probability * hit
    assert expectation == pytest.approx(1., abs=1e-12)
    assert crossed <= .05


@pytest.mark.parametrize('lost,gained', [(0,0),(512,0),(0,512),(25,5),(256,256)])
def test_interval_roots_and_pair_direction(lost,gained):
    n=512
    before=['0']*(n-gained)+['1']*gained
    after=['1']*lost+['0']*(n-lost)
    r=paired_betting_interval(before,after,['0']*n,alpha=.05/3)
    counts=[gained,n-lost-gained,lost]
    assert r['accuracy_drop']==pytest.approx((lost-gained)/n)
    assert r['lower'] <= r['accuracy_drop'] <= r['upper']
    for edge in [r['lower'],r['upper']]:
        if abs(edge)<1:
            assert log_e_value(edge,counts)==pytest.approx(np.log(60),abs=1e-8)


def test_new_schedule_has_no_look_penalty_and_preserves_early_stop():
    r=run_probe(options(),labels=['0']*1500,sample_ids=list(map(str,range(1500))),
                predict=lambda n:(['0']*n,['1']*n,['0']*n))
    assert r['decision']=='expand_models' and len(r['looks'])==1
    assert r['looks'][0]['alpha_at_look']==pytest.approx(.05/3)


def test_legacy_configuration_and_serialization_stay_unchanged():
    raw=options(probe_fraction=.5).model_dump()
    raw.pop('confidence_method')
    old=AdaptivePairedScreeningOptions(**raw)
    assert old.confidence_method=='legacy-exact-v1'
    assert old.model_dump()==raw
    with pytest.raises(ValidationError):
        AdaptivePairedScreeningOptions(**(raw | {'probe_fraction':.75}))


@pytest.mark.parametrize('policy,expected', [('legacy-v1',[]),('candidate-bound-v2',['site'])])
def test_qualitative_routing_retains_only_bound_fields(tmp_path,policy,expected):
    from test_executive_executor import _request,_valid_draft,MockExecutiveClient
    from shortcutminer.experiment.executive_executor import execute_executive
    request=_request(tmp_path)
    request.routing_context.audit_options.qualitative_metadata_policy=policy
    plan=execute_executive(request=request,client=MockExecutiveClient(_valid_draft()))
    qualitative=next(t for t in plan.tasks if t.target=='qualitative')
    assert qualitative.inputs.metadata_fields==expected


def test_bound_metadata_reaches_actual_selected_sample_evidence(tmp_path):
    from test_qualitative_context import _request, _dataset
    from shortcutminer.experiment.qualitative_context import build_qualitative_context
    _dataset(tmp_path)
    request = _request(tmp_path)
    request.task.inputs.metadata_fields = ['site']
    context = build_qualitative_context(request)
    assert context.samples_selected == 4
    assert all(set(sample.metadata) == {'label','site'} for sample in context.selected_samples)


def test_new_confidence_method_in_real_five_agent_cpu_graph(tmp_path,monkeypatch):
    import test_adaptive_screening as fixture
    legacy_options = fixture.options
    monkeypatch.setattr(fixture, 'options', lambda: legacy_options(confidence_method='paired-betting-cs-v1'))
    fixture.test_five_agent_adaptive_path_with_cpu_tensor_fixture(tmp_path,monkeypatch,'strong',None)
    import json
    report=json.loads(next((tmp_path/'runs').glob('*/models/task/*/adaptive_screening.json')).read_text())
    assert report['looks'][0]['method']=='paired_fixed_stake_mixture_cs_v1'
