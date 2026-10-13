"""Bounded paired probes with simultaneous exact-binomial bounds.

Budget decisions are not scientific verdicts. Only the first validation-selected
profile is probed; final evaluation never includes these reserved samples.
"""
import hashlib
import json

import numpy as np
from PIL import Image
from scipy.stats import beta

from shortcutminer.intervention.engine import InterventionEngine
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.task_models.reference import TaskModelError
from shortcutminer.intervention.capabilities import general_image_constructor, supports_adaptive_constructor


def paired_drop_interval(before, after, labels, *, alpha):
    first = np.asarray(before) == np.asarray(labels)
    second = np.asarray(after) == np.asarray(labels)
    if not len(first) or first.shape != second.shape:
        raise ValueError('Matched nonempty predictions required')
    n = len(first)
    # Two binomial intervals, each with half this look's failure allocation.
    def bounds(k):
        tail = alpha / 4
        return (0. if k == 0 else float(beta.ppf(tail, k, n-k+1)),
                1. if k == n else float(beta.ppf(1-tail, k+1, n-k)))
    lost = bounds(int(np.sum(first & ~second)))
    gained = bounds(int(np.sum(~first & second)))
    return {'sample_count': n, 'accuracy_drop': float(np.mean(first.astype(float)-second)),
            'lower': lost[0]-gained[1], 'upper': lost[1]-gained[0],
            'alpha_at_look': alpha, 'method': 'bonferroni_exact_discordance_bounds'}


def budget_decision(interval, delta):
    if interval['upper'] < delta:
        return 'stop_low_effect'
    if interval['lower'] > delta:
        return 'expand_models'
    return 'continue_pairs'


def run_probe(options, *, labels, sample_ids, predict, task_learned=True, baseline_floor=None):
    """predict(n) returns original, intervention, no-op-control predictions."""
    counts = sorted(set(min(n, len(labels)) for n in options.pair_checkpoints))
    if not labels or not counts or counts[0] < 2:
        return {'decision': 'deferred_insufficient_pairs', 'looks': []}
    if not task_learned:
        return {'decision': 'deferred_task_not_learned', 'looks': []}
    sequential = options.confidence_method == 'paired-betting-cs-v1'
    alpha = options.family_alpha / (options.candidate_budget * (1 if sequential else len(counts)))
    looks = []
    for n in counts:
        before, after, control = predict(n)
        if len(before) != n or len(after) != n or len(control) != n:
            raise TaskModelError('Adaptive probe prediction coverage changed')
        if not np.array_equal(before, control):
            return {'decision': 'deferred_processing_control_failed', 'looks': looks}
        if baseline_floor is not None and float(np.mean(np.asarray(before)==np.asarray(labels[:n]))) <= baseline_floor:
            return {'decision': 'deferred_processing_task_not_learned', 'looks': looks}
        if sequential:
            from shortcutminer.experiment.paired_confidence_sequence import paired_betting_interval
            interval = paired_betting_interval(before, after, labels[:n], alpha=alpha)
        else:
            interval = paired_drop_interval(before, after, labels[:n], alpha=alpha)
        decision = budget_decision(interval, options.minimum_effect)
        looks.append({**interval, 'decision': decision,
                      'sensitivity': {str(delta): budget_decision(interval, delta) for delta in options.sensitivity_effects},
                      'sample_ids': sample_ids[:n], 'predictions_original': list(map(str, before)),
                      'predictions_intervention': list(map(str, after)), 'predictions_control': list(map(str, control))})
        if decision != 'continue_pairs':
            return {'decision': decision, 'looks': looks}
    return {'decision': 'deferred_pair_budget_exhausted', 'looks': looks}


def prepare_probe(parameters, access, test, model, directory):
    options = parameters.adaptive_screening
    mode = getattr(parameters.screening_transformation, 'mode', parameters.screening_transformation.type)
    patch = parameters.screening_transformation.type == 'corner_patch'
    provided = parameters.screening_transformation.type == 'provided_pair'
    spatial = parameters.screening_transformation.type in {'spatial_mask','spatial_replace'}
    photometric = general_image_constructor(parameters.screening_transformation)
    canonical = patch and mode.startswith('canonical_')
    if not supports_adaptive_constructor(parameters.screening_transformation):
        raise TaskModelError('Unsupported image constructor for adaptive paired probing')
    def intensity(rgb, source=False):
        if photometric:
            # Use complete SOURCE pixels for duplicate grouping. The transformed
            # pixels are not intensity invariant; labels/instance independence
            # cannot be certified from these hashes.
            return rgb if source else np.asarray(rgb.shape)
        if spatial:
            from shortcutminer.intervention.engine import _region_mask
            return rgb[~_region_mask(parameters.screening_transformation,rgb.shape[1],rgb.shape[0])]
        if provided:
            from shortcutminer.intervention.provided_pairs import allowed_mask
            return rgb[~allowed_mask(rgb, parameters.screening_transformation.changed_region)]
        if patch:
            from shortcutminer.intervention.corner_patch import invariant_content, locate
            if source: locate(rgb,parameters.screening_transformation.patch_size,parameters.screening_transformation.patch_color)
            return invariant_content(rgb,parameters.screening_transformation.patch_size)
        if mode in {'swap_red_green', 'grayscale_max'}:
            # IRM's foreground occupies exactly one of red/green; min(RGB) is
            # identically zero and cannot identify independent digit content.
            if source and (rgb[:, :, 2].any() or np.any((rgb[:, :, 0] > 0) & (rgb[:, :, 1] > 0))):
                raise TaskModelError('swap_red_green requires canonical two-color foreground digits')
            return rgb.max(axis=2)
        return rgb.min(axis=2)
    permutation = [1, 0, 2] if mode == 'swap_red_green' else [2, 0, 1]
    rows = test[test[parameters.split_column].astype(str).isin(parameters.screening_split_values)].copy()
    rows = rows.sort_values(parameters.sample_id_column, kind='stable').reset_index(drop=True)
    groups = {}
    for index, row in rows.iterrows():
        rgb = load_rgb_image_file(access.dataset_file(str(row[parameters.sample_path_column]))).rgb
        digit_hash = hashlib.sha256(intensity(rgb, source=True).tobytes()).hexdigest()
        groups.setdefault(digit_hash, []).append(index)
    # Reserve whole recovered-digit groups; one random representative per group
    # prevents duplicate recolorings from masquerading as independent examples.
    generator = np.random.default_rng(parameters.seed)
    names = sorted(groups)
    order = generator.permutation(len(names))
    reserved = max(0, min(int(len(names)*options.probe_fraction), max(options.pair_checkpoints)))
    chosen = [groups[names[i]] for i in order[:reserved]]
    selected = rows.iloc[[int(generator.choice(indices)) for indices in chosen]].reset_index(drop=True)
    reserved_ids = rows.iloc[[index for indices in chosen for index in indices]][parameters.sample_id_column].astype(str).tolist()
    ids = selected[parameters.sample_id_column].astype(str).tolist()
    labels = selected[parameters.label_column].astype(str).tolist()
    report_path = directory / 'adaptive_screening.json'
    source_paths = [access.dataset_file(str(reference)) for reference in selected[parameters.sample_path_column]]
    identity = {'parameters': parameters.model_dump(mode='json'), 'reserved_probe_ids': reserved_ids,
                'source_hashes': [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths]}
    from shortcutminer.experiment.evidence_promotion import stable_hash
    identity_hash = stable_hash(identity)
    before, after, controls = [], [], []
    evidence = []
    inference_snapshots = []
    def callback(fitted_model, profile):
        # fit_matrix invokes this once, after the complete pilot training protocol.
        if report_path.exists():
            saved = json.loads(report_path.read_text(encoding='utf-8'))
            if saved.get('report_hash') != stable_hash({k:v for k,v in saved.items() if k != 'report_hash'}) or saved['input_hash'] != identity_hash or saved['pilot_weights_sha256'] != profile['weights_sha256']:
                raise TaskModelError('Adaptive resume identity changed')
            for item in saved['pixel_evidence']:
                for prefix in (('changed', 'control', 'anchor') if canonical else ('changed', 'control')):
                    path = (directory / item[prefix+'_reference']).resolve(strict=True)
                    path.relative_to(directory.resolve())
                    if hashlib.sha256(path.read_bytes()).hexdigest() != item[prefix+'_sha256']:
                        raise TaskModelError('Adaptive probe image identity changed')
            model.adaptive_report = saved
            return saved['decision'] == 'expand_models'
        def predict(n):
            for index in range(len(before), n):
                source = source_paths[index]
                original = load_rgb_image_file(source).rgb
                stem = hashlib.sha256(ids[index].encode()).hexdigest()[:20]
                suffix = '.jpg' if parameters.screening_transformation.type == 'compression' else '.png'
                changed = directory / 'probe_images' / (stem+'-changed'+suffix)
                control = directory / 'probe_images' / (stem+'-control.png')
                anchor = directory / 'probe_images' / (stem+'-anchor.png')
                changed.parent.mkdir(parents=True, exist_ok=True)
                quality = InterventionEngine().apply(source_path=source, output_path=changed, transformation=parameters.screening_transformation,
                    dataset_root=access.context.dataset_root)
                transformed = load_rgb_image_file(changed).rgb
                expected = np.repeat(original.max(axis=2, keepdims=True), 3, axis=2) if mode == 'grayscale_max' else original[:, :, permutation]
                if patch:
                    from shortcutminer.intervention.corner_patch import transform
                    expected=transform(original,parameters.screening_transformation)
                if provided:
                    from shortcutminer.intervention.provided_pairs import resolve_pair
                    expected, _ = resolve_pair(access.context.dataset_root, source, parameters.screening_transformation)
                if spatial or photometric:
                    from shortcutminer.intervention.engine import _transform_basic
                    expected = _transform_basic(original, parameters.screening_transformation)
                    if parameters.screening_transformation.type == 'compression':
                        from io import BytesIO
                        buffer=BytesIO()
                        Image.fromarray(expected).save(buffer,format='JPEG',quality=parameters.screening_transformation.quality)
                        buffer.seek(0)
                        with Image.open(buffer) as decoded: expected=np.asarray(decoded.convert('RGB'))
                if not quality.same_dimensions or not np.array_equal(transformed, expected) or not np.array_equal(intensity(original), intensity(transformed)):
                    raise TaskModelError('Adaptive color intervention failed exact pixel invariants')
                baseline = original
                if canonical:
                    from shortcutminer.intervention.corner_patch import matched_control
                    baseline = matched_control(original,parameters.screening_transformation)
                    Image.fromarray(baseline).save(anchor,format='PNG')
                Image.fromarray(baseline).save(control, format='PNG')
                before.append(anchor if canonical else source); after.append(changed); controls.append(control)
                evidence.append({'sample_id': ids[index], 'source_reference': str(selected.iloc[index][parameters.sample_path_column]),
                                 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                 'changed_reference': 'probe_images/'+changed.name, 'changed_sha256': hashlib.sha256(changed.read_bytes()).hexdigest(),
                                 'control_reference': 'probe_images/'+control.name, 'control_sha256': hashlib.sha256(control.read_bytes()).hexdigest()})
                if canonical:
                    evidence[-1].update(anchor_reference='probe_images/'+anchor.name,anchor_sha256=hashlib.sha256(anchor.read_bytes()).hexdigest())
            logits = [model._infer(fitted_model, paths[:n]) for paths in (before, after, controls)]
            if not np.allclose(logits[0], logits[2], rtol=1e-5, atol=1e-5):
                raise TaskModelError('No-op processing control changed model logits')
            inference_snapshots.append({'sample_count':n, 'sample_ids':ids[:n], 'classes':model.classes_.tolist(),
                'logits_original':logits[0].tolist(), 'logits_intervention':logits[1].tolist(), 'logits_control':logits[2].tolist()})
            if canonical:
                raw_logits = model._infer(fitted_model,source_paths[:n])
                raw_predictions = model.classes_[raw_logits.argmax(1)]
                anchor_predictions = model.classes_[logits[0].argmax(1)]
                inference_snapshots[-1].update(logits_raw_source=raw_logits.tolist(),
                    raw_source_accuracy=float(np.mean(raw_predictions==labels[:n])),
                    matched_anchor_accuracy=float(np.mean(anchor_predictions==labels[:n])),
                    preprocessing_accuracy_drop=float(np.mean(raw_predictions==labels[:n])-np.mean(anchor_predictions==labels[:n])))
            return tuple(model.classes_[values.argmax(1)].tolist() for values in logits)
        # Validation is only a task-learning sanity check, not an inferential test.
        validation_labels = model._screening_validation_labels
        majority = max(validation_labels.count(label) for label in set(validation_labels)) / len(validation_labels)
        report = run_probe(options, labels=labels, sample_ids=ids, predict=predict,
                           task_learned=profile['validation_accuracy'] > majority,
                           baseline_floor=majority if canonical else None)
        report.update(input_hash=identity_hash, pilot_weights_sha256=profile['weights_sha256'],
                      options=options.model_dump(mode='json'), reserved_probe_ids=reserved_ids,
                      probe_sample_ids=ids, sampling_unit='exact_content_outside_intervention_region' if spatial or provided else 'exact_content_outside_corner_regions' if patch else 'exact_recovered_digit_content_group',
                      pixel_evidence=evidence, inference_snapshots=inference_snapshots, validation_majority_baseline=majority,
                      comparison_baseline='matched_all_corner_anchor' if canonical else 'raw_source',
                      limitations=['Probe decision concerns this model and intervention only; it is not absence of shortcuts.',
                                   'Intervals condition on this validation-selected pilot model and assume independent held-out digit-content groups.',
                                   'Sensitivity choices are descriptive and do not change the frozen primary budget decision.'])
        if photometric:
            report['sampling_unit'] = 'exact_source_pixels_not_verified_physical_instances'
            report['limitations'] = [value.replace('held-out digit-content groups', 'held-out source groups') for value in report['limitations']]
            report['limitations'].extend(['General image changes do not certify preservation of class-defining content or task labels.',
                                         'Exact pixel uniqueness does not establish instance independence; confidence guarantees require independent source groups.'])
        if canonical:
            report['limitations'].extend(['Both sides replace all corner content; effect is conditional on this matched preprocessing.',
                                         'Raw-to-anchor accuracy changes are preprocessing diagnostics, not patch-dependence evidence.',
                                         'Hash-assigned positions do not guarantee exactly balanced finite-sample label/position counts.'])
        report['report_hash'] = stable_hash(report)
        atomic_write_json(report_path, report, error_type=TaskModelError)
        model.adaptive_report = report
        return report['decision'] == 'expand_models'
    return reserved_ids, callback
