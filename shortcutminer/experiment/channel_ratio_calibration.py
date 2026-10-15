"""Fit a channel-ratio center from positive pooled training means only."""
import hashlib,json
from typing import Literal
import numpy as np
from shortcutminer.protocol.identify import StrictProtocolModel
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.intervention.protocol import ChannelRatioTransform
from shortcutminer.intervention.channel_ratio import CHANNELS
from shortcutminer.experiment.luminance_calibration import training_images

class ChannelRatioCalibrationChoice(StrictProtocolModel):
    mode: Literal['normalize_mean','reflect_log_ratio']
    strategy: Literal['pooled_median','pooled_mean']

def calibrate_channel_ratio(choice,binding,resources,root):
    if binding['kind']!='ratio': raise ValueError('Channel calibration requires a frozen channel-mean ratio')
    numerator=binding['descriptor_key'].removeprefix('rgb_mean_')
    denominator=binding['denominator_key'].removeprefix('rgb_mean_')
    if numerator not in CHANNELS or denominator not in CHANNELS or numerator==denominator:
        raise ValueError('Only a frozen ratio of distinct RGB channel means can use channel-ratio calibration')
    values=[];hashes={}
    for ref,path in training_images(resources,root):
        rgb=load_rgb_image_file(path).rgb
        a=float(rgb[:,:,CHANNELS[numerator]].mean());b=float(rgb[:,:,CHANNELS[denominator]].mean())
        if a<=0 or b<=0:raise ValueError('Training channels include zero means; this log-ratio constructor is unavailable without introducing an unapproved regularizer')
        values.append(np.log(a)-np.log(b));hashes[ref]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not values:raise ValueError('No training images for channel calibration')
    center=float(np.median(values) if choice.strategy=='pooled_median' else np.mean(values))
    operation=ChannelRatioTransform(type='channel_ratio',mode=choice.mode,
        numerator_channel=numerator,denominator_channel=denominator,center_ratio=float(np.exp(center)))
    receipt={'choice':choice.model_dump(mode='json'),'transformation':operation.model_dump(mode='json'),
        'scope':'training_only_label_blind_no_predictions','sample_count':len(values),'image_hashes':hashes,
        'log_ratio_center':center,'training_log_ratio_range':[float(min(values)),float(max(values))],
        'limitations':['Per-image reciprocal channel gains preserve geometry, not task semantics or luminance.',
                       'Quantization and gamut clipping can leave residual target differences or change subject colors.',
                       'A channel-ratio intervention tests a scoped chromatic perturbation, not shortcut causality.']}
    receipt['identity']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
    return receipt
