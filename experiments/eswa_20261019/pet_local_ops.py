"""Prespecified evaluator-only matched-area local transforms and paired summaries."""
import numpy as np
from PIL import Image,ImageFilter
from scipy.ndimage import binary_erosion,distance_transform_edt


def central_mask(region,count):
    flat=np.flatnonzero(region)
    distances=distance_transform_edt(region).ravel()[flat]
    order=np.lexsort((flat,-distances))
    selected=np.zeros(region.shape,dtype=bool)
    selected.ravel()[flat[order[:count]]]=True
    return selected


def local_pairs(rgb,trimap):
    if rgb.shape[:2]!=trimap.shape:raise ValueError('Trimap/image shape mismatch')
    if not set(np.unique(trimap)).issubset({1,2,3}):raise ValueError('Unknown official trimap labels')
    fg=binary_erosion(trimap==1,iterations=3);bg=binary_erosion(trimap==2,iterations=3)
    count=int(min(fg.sum(),bg.sum()))
    if count<64:raise ValueError('Insufficient safe matched-area foreground/background')
    foreground=central_mask(fg,count);background=central_mask(bg,count)
    blurred=np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=3.0)))
    background_pixels=np.where(background[...,None],blurred,rgb)
    foreground_pixels=np.where(foreground[...,None],blurred,rgb)
    if not np.array_equal(background_pixels[trimap!=2],rgb[trimap!=2]):raise ValueError('Background transform altered subject/boundary')
    if not np.array_equal(foreground_pixels[trimap!=1],rgb[trimap!=1]):raise ValueError('Foreground control altered background/boundary')
    noop=rgb.copy()
    return {'noop':noop,'background':background_pixels,'foreground_control':foreground_pixels}, {
        'planned_edit_pixels_each':count,'background_changed_pixels':int(np.any(background_pixels!=rgb,axis=2).sum()),
        'foreground_changed_pixels':int(np.any(foreground_pixels!=rgb,axis=2).sum()),
        'image_pixels':int(trimap.size),'subject_boundary_unchanged_for_background':True,
        'background_boundary_unchanged_for_foreground_control':True,'noop_exact_pixels':True,
        'matched_area_note':'Planned eligible mask cardinality is matched; effective changed pixels may differ because locally flat pixels remain unchanged.'}


def paired_effects(correct,iterations=5000,seed=20261020):
    baseline=np.asarray(correct['original'],dtype=float)
    if not len(baseline):raise ValueError('No paired outcomes')
    bg=baseline-np.asarray(correct['background'],dtype=float)
    fg=baseline-np.asarray(correct['foreground_control'],dtype=float)
    diffs={'background_accuracy_drop':bg,'foreground_control_accuracy_drop':fg,
        'background_minus_foreground_drop':bg-fg}
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(baseline),size=(iterations,len(baseline)))
    output={}
    for key,difference in diffs.items():
        draws=difference[indices].mean(axis=1)
        output[key]={'estimate':float(difference.mean()),'pointwise_95_interval':np.quantile(draws,[0.025,0.975]).tolist(),
            'bonferroni_three_contrast_interval':np.quantile(draws,[0.05/6,1-0.05/6]).tolist(),
            'paired_difference_counts':{str(value):int((difference==value).sum()) for value in np.unique(difference)}}
    return output
