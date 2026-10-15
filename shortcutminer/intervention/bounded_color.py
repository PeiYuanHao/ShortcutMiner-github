"""Preserve float RGB luminance while compressing chroma into the valid gamut."""
import numpy as np
from shortcutminer.intervention.channel_ratio import CHANNELS


def transform(rgb, spec):
    a=rgb.astype(np.float64)
    n=CHANNELS[spec.numerator_channel];d=CHANNELS[spec.denominator_channel]
    first=float(a[:,:,n].mean());second=float(a[:,:,d].mean())
    if first<=0 or second<=0: raise ValueError('Positive channel means required')
    current=np.log(first/second);center=np.log(spec.center_ratio)
    desired=center if spec.mode=='normalize_mean' else 2*center-current
    shift=np.clip(desired-current,-spec.max_log_shift,spec.max_log_shift)
    b=a.copy();b[:,:,n]*=np.exp(shift/2);b[:,:,d]*=np.exp(-shift/2)
    weights=np.array([.299,.587,.114]);y=a@weights
    chroma=b-(b@weights)[:,:,None]
    positive=np.where(chroma>0,chroma,1);negative=np.where(chroma<0,-chroma,1)
    upper=np.where(chroma>0,(255-y[:,:,None])/positive,np.inf)
    lower=np.where(chroma<0,y[:,:,None]/negative,np.inf)
    scale=np.minimum(1,np.minimum(upper,lower).min(axis=2))
    result=y[:,:,None]+scale[:,:,None]*chroma
    return np.clip(np.rint(result),0,255).astype(np.uint8)
