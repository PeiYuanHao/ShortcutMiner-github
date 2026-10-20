import numpy as np
from experiments.eswa_20261019.prepare_inputs import transform
from experiments.eswa_20261019.synthetic_specificity import exact_pairs


def test_exact_oracle_restores_pristine_and_does_not_touch_noncue_pixels():
    raw=np.random.default_rng(9).integers(0,256,(32,32,3),dtype=np.uint8)
    geometry={'watermark_xy':[6,29],'patch_xy':[[5,5],[31,31]]}
    for mechanism in ('border','watermark','position_patch','no_artificial_cue'):
        for state in (0,1):
            original,support=transform(raw,mechanism,state,geometry)
            images,fidelity=exact_pairs(original,raw,{'state':state},{'spec':{'mechanism':mechanism},'geometry':geometry})
            assert np.array_equal(images['removed'][4:36,4:36],raw)
            assert np.array_equal(images['removed'][~support],original[~support])
            assert np.array_equal(images['noop'],original)
            assert fidelity['oracle_noncue_pixels_changed']==0
