import numpy as np
import pytest
from shortcutminer.experiment.prediction_diagnostics import paired_score_diagnostics


def test_margin_response_without_accuracy_flip_and_reordered_multiclass_labels():
    before=np.array([[0.,5.,1.],[6.,1.,2.]])
    after=np.array([[0.,3.,1.],[4.,1.,2.]])
    assert np.array_equal(before.argmax(1),after.argmax(1))
    result=paired_score_diagnostics(before,after,['cat','dog'],['dog','cat','other'])
    assert result['original_true_class_margin_mean']==4
    assert result['counterfactual_true_class_margin_mean']==2
    assert result['logit_margin_delta']==-2
    assert result['true_class_margin_decreased_fraction']==1


def test_nonfinite_or_mismatched_score_evidence_is_rejected():
    with pytest.raises(ValueError):paired_score_diagnostics([[0,np.nan]],[[0,1]],['x'],['x','y'])
    with pytest.raises(ValueError):paired_score_diagnostics([[0,1]],[[0,1],[0,1]],['x'],['x','y'])
