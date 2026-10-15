"""Training-only preview coverage and opt-in semantic review contracts."""
import random
from typing import Literal
import numpy as np
from pydantic import Field
from shortcutminer.protocol.identify import StrictProtocolModel,NonEmptyText
from shortcutminer.experiment.channel_ratio_calibration import ChannelRatioCalibrationChoice,calibrate_channel_ratio
from shortcutminer.experiment.luminance_calibration import training_images
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.intervention.protocol import BoundedChannelRatioTransform

POLICY='training-construction-review-v2'


class BoundedRatioChoice(ChannelRatioCalibrationChoice):
    max_log_shift: float = Field(gt=0,le=1.3862943611198906)


def calibrate_bounded_ratio(choice,contract,resources,root):
    import hashlib,json
    old=calibrate_channel_ratio(choice,contract,resources,root)
    old.pop('identity')
    old['choice']=choice.model_dump(mode='json')
    old['transformation']=BoundedChannelRatioTransform.model_validate({
        **old['transformation'],'type':'bounded_channel_ratio','max_log_shift':choice.max_log_shift}).model_dump(mode='json')
    old['limitations']=['Float RGB luminance is preserved per pixel before 8-bit rounding; geometry is unchanged.',
        'Gamut compression and bounded shifts can leave residual channel-ratio information.',
        'Subject colors change with background colors; this is not illumination/foreground isolation or semantic certification.']
    old['identity']=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()
    return old


def preview_sources(operation,resources,root):
    """8 reproducible random images + 4 target extremes; no label/outcome selection."""
    sources=training_images(resources,root)
    if len(sources)<=12:return sources
    random_rows=random.Random(2026).sample(sources,8)
    ranked=[]
    for ref,path in sources:
        a=load_rgb_image_file(path).rgb
        if operation.type in {'channel_ratio','bounded_channel_ratio'}:
            from shortcutminer.intervention.channel_ratio import CHANNELS
            n=a[:,:,CHANNELS[operation.numerator_channel]].mean();d=a[:,:,CHANNELS[operation.denominator_channel]].mean()
            value=float(n/d) if d>0 else float('inf')
        else:
            value=float((a@np.array([.299,.587,.114])).mean())
        ranked.append((value,ref,path))
    ranked.sort(key=lambda row:(row[0],row[1]));used={r[0] for r in random_rows}
    extremes=[]
    for target,pool in ((2,ranked),(4,list(reversed(ranked)))):
        for _,ref,path in pool:
            if ref not in used: extremes.append((ref,path));used.add(ref)
            if len(extremes)>=target:break
    return random_rows+extremes


def pixel_diagnostics(before,after):
    a=before.astype(float);b=after.astype(float);weights=np.array([.299,.587,.114])
    return {'new_rgb_boundary_fraction':float((((b==0)|(b==255))&((a>0)&(a<255))).mean()),
        'mean_absolute_rgb_change':float(np.abs(a-b).mean()),
        'mean_absolute_pixel_luminance_change':float(np.abs(a@weights-b@weights).mean()),
        'maximum_absolute_pixel_luminance_change':float(np.abs(a@weights-b@weights).max()),
        'mean_absolute_horizontal_gradient_change':float(np.abs(np.diff(a@weights,axis=1)-np.diff(b@weights,axis=1)).mean()) if a.shape[1]>1 else 0.0}
