import numpy as np
import pytest
from experiments.eswa_20261019.pet_local_ops import local_pairs,paired_effects


def test_local_pairs_match_masks_and_protect_other_regions():
    rng=np.random.default_rng(7);rgb=rng.integers(0,256,(64,64,3),dtype=np.uint8)
    trimap=np.full((64,64),2,dtype=np.uint8);trimap[12:52,12:52]=3;trimap[14:50,14:50]=1
    images,fidelity=local_pairs(rgb,trimap)
    assert np.array_equal(images['noop'],rgb)
    assert np.array_equal(images['background'][trimap!=2],rgb[trimap!=2])
    assert np.array_equal(images['foreground_control'][trimap!=1],rgb[trimap!=1])
    assert fidelity['planned_edit_pixels_each']>=64
    assert np.any(images['background']!=rgb) and np.any(images['foreground_control']!=rgb)
    assert fidelity['background_changed_pixels']==fidelity['foreground_changed_pixels']


def test_unknown_or_tiny_annotation_regions_defer():
    rgb=np.zeros((20,20,3),dtype=np.uint8)
    with pytest.raises(ValueError):local_pairs(rgb,np.zeros((20,20),dtype=np.uint8))
    trimap=np.full((20,20),2,dtype=np.uint8);trimap[9:11,9:11]=1
    with pytest.raises(ValueError):local_pairs(rgb,trimap)


def test_bootstrap_preserves_pairing_and_signed_effects():
    result=paired_effects({'original':[True]*8,'background':[False]*8,'foreground_control':[True]*8},iterations=100)
    assert result['background_accuracy_drop']['estimate']==1
    assert result['background_accuracy_drop']['pointwise_95_interval']==[1,1]
    assert result['foreground_control_accuracy_drop']['estimate']==0
    assert result['background_minus_foreground_drop']['estimate']==1
