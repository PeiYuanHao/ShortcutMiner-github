"""Training-only OOD construction diagnostics, never outcome-driven selection."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.intervention.corner_patch import boxes, invariant_content, matched_control, transform
from shortcutminer.intervention.protocol import CornerPatchTransform, BackgroundReplaceTransform, SpatialTransform


def construction_quality(operation, resources, root, *, target_box=None):
    """Structural tests qualify a tool, not its causal interpretation or effect size."""
    if isinstance(operation, SpatialTransform):
        from shortcutminer.experiment.spatial_construction_quality import spatial_quality, training_samples
        return spatial_quality(operation, training_samples(resources, root), target_box=target_box)
    if isinstance(operation, CornerPatchTransform):
        if not operation.mode.startswith('canonical_'):
            return {'status': 'failed', 'checks': {'source_location_trace_removed': False},
                    'reason': 'Only the source corner is filled. Its location can remain visible after removal or relocation. Use a matched all-corner preprocessing recipe or defer.',
                    'scope': 'training_only_no_task_model_predictions'}
        frame = pd.read_csv(Path(root)/resources.model_metadata_file, dtype=str, keep_default_na=False)
        train = frame[frame[resources.split_column].isin(resources.train_split_values)]
        checks = []
        for _, row in train.sort_values(resources.sample_path_column).head(32).iterrows():
            path = (Path(root)/row[resources.sample_path_column]).resolve(strict=True)
            path.relative_to(Path(root).resolve())
            rgb = load_rgb_image_file(path).rgb
            control, changed = matched_control(rgb, operation), transform(rgb, operation)
            corners = boxes(rgb, operation.patch_size)
            valid = (np.array_equal(invariant_content(rgb, operation.patch_size), invariant_content(control, operation.patch_size))
                     and np.array_equal(invariant_content(control, operation.patch_size), invariant_content(changed, operation.patch_size)))
            for x0,y0,x1,y1 in corners:
                block = changed[y0:y1,x0:x1]
                valid &= bool(np.all(block == operation.fill_color) or
                              (operation.mode == 'canonical_randomize_position' and np.all(block == operation.patch_color)))
            # Change the hypothetical source location, keep protected content fixed:
            # counterfactual output must not disclose which corner originally held the patch.
            neutral = transform(rgb, operation.model_copy(update={'mode': 'canonical_remove'}))
            variants = []
            for x0,y0,x1,y1 in corners:
                source = neutral.copy(); source[y0:y1,x0:x1] = operation.patch_color
                variants.append(transform(source, operation))
            valid &= all(np.array_equal(variants[0], item) for item in variants[1:])
            checks.append({'source_reference': row[resources.sample_path_column],
                           'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'passed': bool(valid)})
        passed = bool(checks) and all(item['passed'] for item in checks)
        return {'status': 'passed' if passed else 'failed', 'samples': checks,
                'checks': {'protected_pixels_identical': passed, 'source_location_trace_removed': passed,
                           'matched_processing_control_required': True},
                'comparison_baseline': 'all_corner_processed_original_patch_position',
                'scope': 'training_only_no_task_model_predictions',
                'limitations': ['Original content beneath patches is unavailable; four corner regions are replaced on both sides.',
                               'Structural cue erasure does not independently certify label semantics.']}
    if isinstance(operation, BackgroundReplaceTransform):
        from shortcutminer.intervention.local_foreground import LocalForegroundSegmentationProvider
        provider = LocalForegroundSegmentationProvider(operation.local_foreground)
        frame = pd.read_csv(Path(root)/resources.model_metadata_file, dtype=str, keep_default_na=False)
        train = frame[frame[resources.split_column].isin(resources.train_split_values)]
        samples = []
        try:
            for _, row in train.sort_values(resources.sample_path_column).head(8).iterrows():
                path = (Path(root)/row[resources.sample_path_column]).resolve(strict=True)
                path.relative_to(Path(root).resolve())
                try:
                    segmented = provider.segment(image_path=path, mask_path=None,
                                                 max_file_bytes=50_000_000, max_pixels=25_000_000)
                    rgb = load_rgb_image_file(path).rgb
                    changed = rgb.copy(); changed[~segmented.mask] = operation.fill_color
                    passed = np.array_equal(rgb[segmented.mask], changed[segmented.mask]) and np.any(rgb[~segmented.mask] != changed[~segmented.mask])
                    samples.append({'source_reference': row[resources.sample_path_column], 'passed': bool(passed), **provider.last_details})
                except Exception as exc:
                    samples.append({'source_reference': row[resources.sample_path_column], 'passed': False,
                                    'failure': type(exc).__name__ + ': ' + str(exc)})
        finally:
            provider.close()
        usable = sum(item['passed'] for item in samples)
        return {'status': 'passed' if usable else 'failed', 'samples': samples, 'usable_count': usable,
                'failed_count': len(samples)-usable, 'scope': 'training_only_local_mask_preview',
                'checks': {'protected_foreground_pixels_identical': bool(usable)},
                'limitations': ['Mask errors and failures must be retained; held-out images require the same per-image checks.',
                                'Predicted segmentation cannot guarantee task-label preservation.']}
    return {'status': 'passed', 'scope': 'training_compatibility_checked',
            'limitations': ['Color recipes still require exact per-image intensity invariants during execution.']}
