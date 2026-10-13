"""Label-blind training calibration. It never opens validation/ID images."""
import hashlib
import json
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from PIL import Image
from pydantic import Field
from shortcutminer.protocol.identify import StrictProtocolModel
from shortcutminer.intervention.protocol import LuminanceTransform
from shortcutminer.inspection.image import load_rgb_image_file

class LuminanceCalibrationChoice(StrictProtocolModel):
    mode: Literal['normalize_mean','reflect_mean']
    strategy: Literal['pooled_median','pooled_mean','mid_quantiles']
    lower_quantile: float = Field(default=.1,ge=0,lt=.5)
    upper_quantile: float = Field(default=.9,gt=.5,le=1)

def training_images(resources, root, *, limit=None):
    root=Path(root).resolve(strict=True)
    metadata=(root/resources.model_metadata_file).resolve(strict=True);metadata.relative_to(root)
    if not resources.train_split_values or not resources.split_column:
        raise ValueError('Explicit authoritative training split required')
    frame=pd.read_csv(metadata,dtype=str,keep_default_na=False,nrows=100001)
    if len(frame)>100000:raise ValueError('Construction metadata exceeds bounded row budget')
    fields=[resources.sample_path_column,resources.split_column]
    if any(k is None or k not in frame for k in fields):raise ValueError('Missing authoritative construction columns')
    train=frame[frame[resources.split_column].isin(resources.train_split_values)]
    count=min(getattr(resources,'max_train_samples',None) or 10000,10000)
    if limit is not None:count=min(count,limit)
    if train.empty:raise ValueError('No training images available')
    result=[]
    for ref in sorted(set(train[resources.sample_path_column]))[:count]:
        path=(root/ref).resolve(strict=True);path.relative_to(root)
        result.append((ref,path))
    return result

def calibrate_luminance(choice,resources,root):
    choice=LuminanceCalibrationChoice.model_validate(choice)
    values=[];hashes={}
    for ref,path in training_images(resources,root):
        rgb=load_rgb_image_file(path).rgb
        values.append(float(np.asarray(Image.fromarray(rgb).convert('YCbCr'))[:,:,0].mean()))
        hashes[ref]=hashlib.sha256(path.read_bytes()).hexdigest()
    if choice.strategy=='pooled_median':center=float(np.median(values))
    elif choice.strategy=='pooled_mean':center=float(np.mean(values))
    else:center=float(np.mean(np.quantile(values,[choice.lower_quantile,choice.upper_quantile])))
    transform=LuminanceTransform(type='luminance',mode=choice.mode,
        **({'target_mean':center} if choice.mode=='normalize_mean' else {'pivot_mean':center}))
    receipt={'choice':choice.model_dump(mode='json'),'transformation':transform.model_dump(mode='json'),
        'scope':'training_only_label_blind_no_predictions','sample_count':len(values),'image_hashes':hashes,
        'pooled_y_mean':float(np.mean(values)),'pooled_y_quantiles':np.quantile(values,[0,.1,.5,.9,1]).tolist(),
        'limitations':['YCbCr/RGB quantization and gamut clipping can alter chroma or contrast.',
                      'Neutralizing or reflecting exposure does not guarantee preservation of task semantics.']}
    receipt['identity']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
    return receipt
