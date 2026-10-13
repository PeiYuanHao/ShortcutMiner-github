"""Patch changes stay within verified geometry and never depend on labels."""
import numpy as np
import pytest
from PIL import Image
from shortcutminer.intervention.corner_patch import boxes,locate,invariant_content,transform
from shortcutminer.intervention.protocol import CornerPatchTransform
from shortcutminer.intervention.engine import InterventionEngine

@pytest.mark.parametrize('mode',['remove','randomize_position','canonical_remove','canonical_randomize_position'])
@pytest.mark.parametrize('corner',range(4))
def test_actual_engine_preserves_protected_pixels_and_exact_template(tmp_path,mode,corner):
    rgb=np.random.default_rng(42).integers(0,240,(32,32,3),dtype=np.uint8)
    x0,y0,x1,y1=boxes(rgb,4)[corner];rgb[y0:y1,x0:x1]=[255,0,0]
    spec=CornerPatchTransform(type='corner_patch',mode=mode,patch_size=4,patch_color=[255,0,0],seed=42)
    assert locate(rgb,4,spec.patch_color)==corner
    source=tmp_path/'source.png';target=tmp_path/'target.png';Image.fromarray(rgb).save(source)
    quality=InterventionEngine().apply(source_path=source,output_path=target,transformation=spec)
    changed=np.array(Image.open(target))
    assert quality.same_dimensions and np.array_equal(changed,transform(rgb,spec))
    assert np.array_equal(invariant_content(rgb,4),invariant_content(changed,4))
    assert np.array_equal(changed,transform(rgb,spec))
    if mode in {'randomize_position','canonical_randomize_position'}: assert locate(changed,4,spec.patch_color) in range(4)
    else:
        with pytest.raises(ValueError,match='Exactly one'): locate(changed,4,spec.patch_color)

def test_ambiguous_or_missing_patch_is_not_silently_processed():
    spec=CornerPatchTransform(type='corner_patch',mode='remove',patch_size=4,patch_color=[255,0,0])
    for rgb in [np.zeros((32,32,3),dtype=np.uint8),np.full((32,32,3),[255,0,0],dtype=np.uint8)]:
        with pytest.raises(ValueError,match='Exactly one'): transform(rgb,spec)


@pytest.mark.parametrize('mode',['canonical_remove','canonical_randomize_position'])
def test_canonical_counterfactual_hides_source_position_with_matched_anchor(mode):
    from shortcutminer.intervention.corner_patch import matched_control
    content=np.random.default_rng(71).integers(0,230,(32,32,3),dtype=np.uint8)
    spec=CornerPatchTransform(type='corner_patch',mode=mode,patch_size=4,patch_color=[255,0,0],seed=42)
    changed=[];anchors=[]
    for x0,y0,x1,y1 in boxes(content,4):
        source=content.copy();source[y0:y1,x0:x1]=spec.patch_color
        changed.append(transform(source,spec));anchors.append(matched_control(source,spec))
    assert all(np.array_equal(changed[0],item) for item in changed)
    assert [locate(anchor,4,spec.patch_color) for anchor in anchors]==list(range(4))
    assert all(np.array_equal(invariant_content(content,4),invariant_content(item,4)) for item in changed+anchors)
