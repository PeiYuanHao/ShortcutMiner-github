from experiments.eswa_20261019.export_analysis import paired_effect,changed_text_leaves,usage_records
import json


def test_reused_development_invocation_is_not_billed_twice(tmp_path):
    calls=tmp_path/'run/model_calls';calls.mkdir(parents=True)
    (calls/'record.json').write_text(json.dumps({'invocation_id':'original-call','status':'completed',
        'usage':{'available':True,'total_tokens':6603}}))
    assert usage_records(tmp_path)[0]==6603
    assert usage_records(tmp_path,{'original-call'})==(0,[],[])


def test_signed_paired_interval_flips_both_bounds_correctly():
    result=paired_effect({'parameters':{'comparison_mode':'paired'},'metrics':{'accuracy_delta':-0.2,'pair_count':500},
        'uncertainty':[{'metric':'accuracy_delta','available':True,'lower':-0.3,'upper':-0.1,'method':'percentile_bootstrap'}]})
    assert result['accuracy_drop']==0.2 and result['lower_95']==0.1 and result['upper_95']==0.3
    assert paired_effect({'parameters':{'comparison_mode':'paired'},'metrics':{}}) is None
    assert paired_effect({'parameters':{'comparison_mode':'independent'},'metrics':{'accuracy_delta':-0.2}}) is None


def test_empty_feedback_does_not_count_as_effective_mask():
    assert changed_text_leaves({'a':[],'b':''},{'a':[],'b':''})==0
    assert changed_text_leaves({'a':['real finding']},{'a':['withheld']})==1
