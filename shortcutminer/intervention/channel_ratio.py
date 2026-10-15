"""Deterministic channel-mean ratio interventions, without labels or predictions."""
import numpy as np

CHANNELS={'red':0,'green':1,'blue':2}

def transform(rgb,spec):
    a=rgb.astype(np.float64)
    numerator=CHANNELS[spec.numerator_channel];denominator=CHANNELS[spec.denominator_channel]
    first=float(a[:,:,numerator].mean());second=float(a[:,:,denominator].mean())
    if first<=0 or second<=0:
        raise ValueError('Channel-ratio intervention requires positive means in both channels; no epsilon or invented color is substituted')
    current=np.log(first)-np.log(second);center=np.log(spec.center_ratio)
    desired=center if spec.mode=='normalize_mean' else 2*center-current
    gain=np.exp(.5*(desired-current))
    if not np.isfinite(gain) or gain<=0: raise ValueError('Nonfinite channel ratio gain')
    a[:,:,numerator]*=gain
    a[:,:,denominator]/=gain
    return np.clip(np.rint(a),0,255).astype(np.uint8)
