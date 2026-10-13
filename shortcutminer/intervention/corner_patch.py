"""Verified constant corner patches; preserve all pixels outside corner regions."""
import hashlib
import numpy as np

CORNERS = ('top_left', 'top_right', 'bottom_left', 'bottom_right')

def boxes(rgb, size):
    h,w=rgb.shape[:2]
    if size*2>=min(h,w): raise ValueError('Corner patch overlaps image center')
    return [(0,0,size,size),(w-size,0,w,size),(0,h-size,size,h),(w-size,h-size,w,h)]

def invariant_content(rgb,size):
    result=rgb.copy()
    for x0,y0,x1,y1 in boxes(rgb,size): result[y0:y1,x0:x1]=0
    return result

def locate(rgb,size,color):
    found=[i for i,(x0,y0,x1,y1) in enumerate(boxes(rgb,size))
           if np.all(rgb[y0:y1,x0:x1]==color)]
    if len(found)!=1: raise ValueError('Exactly one verified constant corner patch is required')
    return found[0]

def transform(rgb,spec):
    source=locate(rgb,spec.patch_size,spec.patch_color)
    output=rgb.copy(); positions=boxes(rgb,spec.patch_size)
    if spec.mode.startswith('canonical_'):
        for x0,y0,x1,y1 in positions:
            output[y0:y1,x0:x1]=spec.fill_color
        if spec.mode == 'canonical_remove':
            return output
    x0,y0,x1,y1=positions[source]; output[y0:y1,x0:x1]=spec.fill_color
    if spec.mode in {'randomize_position','canonical_randomize_position'}:
        key=str(spec.seed).encode()+invariant_content(rgb,spec.patch_size).tobytes()
        target=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%4
        x0,y0,x1,y1=positions[target];output[y0:y1,x0:x1]=spec.patch_color
    if not np.array_equal(invariant_content(rgb,spec.patch_size),invariant_content(output,spec.patch_size)):
        raise ValueError('Corner intervention changed protected content')
    return output


def matched_control(rgb, spec):
    """Same corner processing as canonical OOD, with patch at original position."""
    if not spec.mode.startswith('canonical_'):
        return rgb.copy()
    source = locate(rgb, spec.patch_size, spec.patch_color)
    result = rgb.copy()
    positions = boxes(rgb, spec.patch_size)
    for x0,y0,x1,y1 in positions:
        result[y0:y1,x0:x1] = spec.fill_color
    x0,y0,x1,y1 = positions[source]
    result[y0:y1,x0:x1] = spec.patch_color
    return result
