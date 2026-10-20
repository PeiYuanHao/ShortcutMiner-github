from experiments.eswa_20261019.seed_replication import expansion_allowed,paired_interval


def test_no_extra_profiles_to_rescue_deferred_or_negative_probe():
    assert expansion_allowed('expand_models')
    for decision in ('stop_low_effect','deferred_pair_budget_exhausted','deferred_task_not_learned',None):
        assert not expansion_allowed(decision)


def test_seed_summary_keeps_signed_paired_difference():
    result=paired_interval([True]*5,[False]*5)
    assert result['accuracy_drop']==1 and result['paired_bootstrap_95_interval']==[1,1]
