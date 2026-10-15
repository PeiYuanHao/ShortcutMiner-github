"""Opt-in training-only checks that an operation targets its frozen measurement.

These are execution checks, not evidence that a cue is a shortcut or that labels
are preserved. They use no labels, held-out images, predictions or effect gates.
"""
import math
from shortcutminer.experiment.shortcut_knowledge import AXES
from shortcutminer.inspection.image import inspect_image_file, flatten_image_descriptors

VERSION = 'measurement-intervention-alignment-v1'
DESCRIPTORS = ['dimensions','intensity','rgb_stats','saturation','entropy','edge_density','spatial_stats']
OPERATIONS = {
    'chromaticity': ['global_color','channel_ratio'],
    'global_photometry': ['luminance','brightness','contrast'],
    'texture_structure': ['blur','frequency_filter','noise','compression','resolution'],
    'spatial_localization': ['spatial_mask','spatial_replace'],
    'acquisition_encoding': ['compression','resolution'],
}


def binding_contract(spec):
    if spec.get('status') != 'measurable' or spec.get('kind') not in {'descriptor','ratio'}:
        raise ValueError('Alignment requires a frozen measurable candidate')
    keys = [spec['descriptor_key']]
    if spec['kind']=='ratio': keys.append(spec['denominator_key'])
    axes = [name for name,match in AXES.items() if all(match(key) for key in keys)]
    axis = axes[0] if len(axes)==1 else 'mixed_or_unclassified'
    allowed=OPERATIONS.get(axis)
    rgb_mean_ratio=(spec['kind']=='ratio' and all(key in {'rgb_mean_red','rgb_mean_green','rgb_mean_blue'} for key in keys)
                    and len(set(keys))==2)
    if axis=='chromaticity' and not rgb_mean_ratio:
        allowed=['global_color']
    return {'policy_version':VERSION, 'candidate_id':spec['candidate_id'],
            'kind':spec['kind'], 'descriptor_key':spec['descriptor_key'],
            'denominator_key':spec.get('denominator_key'), 'measurement_axis':axis,
            'eligible_operation_types':allowed,
            'scope':'Execution targeting only; no shortcut verdict or semantic certification. Unknown or mixed axes require measured change and explicit review, not an invented mapping.'}


def validate_operation(contract, candidate_id, descriptor_key, operation):
    key = contract['descriptor_key'] if contract['kind']=='descriptor' else f"ratio:{contract['descriptor_key']}/{contract['denominator_key']}"
    if candidate_id != contract['candidate_id'] or descriptor_key != key:
        raise ValueError('Operation must bind the same frozen candidate and primary measurement')
    allowed = contract['eligible_operation_types']
    operation_type = 'channel_ratio' if operation['type']=='bounded_channel_ratio' else operation['type']
    if allowed is not None and operation_type not in allowed:
        raise ValueError(f"Target-operation mismatch: {contract['measurement_axis']} measured by {key} cannot use {operation['type']} as its primary diagnostic. Choose a compatible supported operation or defer; incidental descriptor changes do not validate this mechanism.")
    if operation_type=='channel_ratio':
        if (contract['kind']!='ratio' or contract['descriptor_key']!='rgb_mean_'+operation['numerator_channel']
                or contract['denominator_key']!='rgb_mean_'+operation['denominator_channel']):
            raise ValueError('Channel-ratio operation must target the exact frozen channel pair')
    if contract['measurement_axis']=='chromaticity' and operation.get('mode')=='channel_scale':
        scales = operation.get('channel_scale', [1,1,1])
        if len(set(scales))==1:
            raise ValueError('Equal channel scales manipulate brightness, not relative color; choose a chromatic intervention or defer')


def _value(path, contract):
    result = inspect_image_file(path, requested_descriptors=DESCRIPTORS,
        max_file_bytes=25000000,max_pixels=20000000,max_frames=1)
    facts = flatten_image_descriptors(result.descriptors,file_size_bytes=result.byte_size)
    value = facts.get(contract['descriptor_key'])
    denominator = facts.get(contract['denominator_key']) if contract['kind']=='ratio' else 1
    if value is None or denominator is None or denominator==0: return None
    value = value/denominator
    return float(value) if math.isfinite(value) else None


def measure_preview_target(contract, records, root, directory):
    """Measure the actual frozen primary target, not a generic pixel-change score."""
    pairs = []
    for record in records:
        source=(root/record['source_reference']).resolve(strict=True); source.relative_to(root.resolve())
        changed=(directory/record['changed_reference']).resolve(strict=True); changed.relative_to(directory.resolve())
        pairs.append({'pair_id':record['pair_id'], 'before':_value(source,contract),'after':_value(changed,contract)})
    valid=[p for p in pairs if p['before'] is not None and p['after'] is not None]
    changed=[p for p in valid if p['before'] != p['after']]
    return {'policy_version':VERSION,'binding':contract,'scope':'training_preview_only',
            'pairs':pairs,'valid_pair_count':len(valid),'changed_target_pair_count':len(changed),
            'status':'measured_change' if changed else 'no_measured_target_change',
            'limitations':['Change is assessed at the registered descriptor precision; no minimum scientific effect is imposed.',
                           'Preview change does not establish label preservation, removal of the mechanism, model reliance, or coverage outside these preview images.']}
