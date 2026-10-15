"""Shared renderer/paired-runtime capability checks, never shortcut conclusions."""
GENERAL_IMAGE_TYPES = frozenset({'brightness','contrast','luminance','channel_ratio','bounded_channel_ratio','rotation','frequency_filter','blur','noise','compression','resolution'})
LEGACY_COLOR_MODES = frozenset({'cycle_rgb','swap_red_green','grayscale_max'})

def general_image_constructor(spec):
    return spec.type in GENERAL_IMAGE_TYPES or (spec.type == 'global_color' and spec.mode not in LEGACY_COLOR_MODES)

def supports_adaptive_constructor(spec):
    return (general_image_constructor(spec) or spec.type in {'spatial_mask','spatial_replace','provided_pair'} or
            (spec.type == 'global_color' and spec.mode in LEGACY_COLOR_MODES) or
            (spec.type == 'corner_patch' and spec.mode in {'remove','randomize_position','canonical_remove','canonical_randomize_position'}))
