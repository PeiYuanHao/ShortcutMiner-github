"""Training-only spatial construction acceptance; never searches model outcomes."""
from pathlib import Path
from typing import Literal
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import Field, model_validator

from shortcutminer.agent_client import invoke_agent_client
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.intervention.engine import _region_mask, _transform_basic
from shortcutminer.intervention.protocol import NormalizedBox, SpatialTransform
from shortcutminer.protocol.identify import CandidateId, NonEmptyText, StrictProtocolModel
from shortcutminer.experiment.ood_recipe_selection_quality import digest


class SpatialConstructionProposal(StrictProtocolModel):
    status: Literal['selected', 'deferred']
    candidate_id: CandidateId
    descriptor_key: NonEmptyText
    target_box: NormalizedBox | None
    target_geometry: Literal['rectangle', 'exterior'] = 'rectangle'
    transformation: SpatialTransform | None
    rationale: NonEmptyText
    intended_invariants: list[NonEmptyText] = Field(min_length=1)
    semantic_risks: list[NonEmptyText] = Field(min_length=1)
    processing_controls: list[NonEmptyText] = Field(min_length=1)
    limitations: list[NonEmptyText] = Field(min_length=1)

    @model_validator(mode='after')
    def selected_requires_localization(self):
        if self.status == 'selected' and (self.target_box is None or self.transformation is None):
            raise ValueError('Selected construction requires a training-localized target and operation')
        if self.status == 'deferred' and (self.target_box is not None or self.transformation is not None):
            raise ValueError('Deferred construction must not bind an operation or target')
        return self


class TrainingLocalization(StrictProtocolModel):
    candidate_id: CandidateId
    descriptor_key: NonEmptyText
    target_box: NormalizedBox
    target_geometry: Literal['rectangle', 'exterior'] = 'rectangle'
    rationale: NonEmptyText


def training_samples(resources, root, *, limit=32):
    root = Path(root).resolve(strict=True)
    metadata = (root / resources.model_metadata_file).resolve(strict=True)
    metadata.relative_to(root)
    frame = pd.read_csv(metadata, dtype=str, keep_default_na=False)
    train = frame[frame[resources.split_column].isin(resources.train_split_values)]
    if train.empty:
        raise ValueError('No training construction samples')
    samples = []
    for reference in train.sort_values(resources.sample_path_column)[resources.sample_path_column].head(limit):
        path = (root / reference).resolve(strict=True)
        path.relative_to(root)
        samples.append((reference, hashlib.sha256(path.read_bytes()).hexdigest(), load_rgb_image_file(path).rgb))
    return samples


def uniform_edge_regions(rgb):
    """Exact connected constant-color border runs, measured from public pixels."""
    h, w = rgb.shape[:2]
    regions = []
    for side in ('top', 'bottom', 'left', 'right'):
        lines = rgb if side in ('top', 'bottom') else rgb.transpose(1, 0, 2)
        if side in ('bottom', 'right'):
            lines = lines[::-1]
        color = lines[0, 0]
        n = 0
        for line in lines:
            if not np.all(line == color):
                break
            n += 1
        if n < 2:
            continue
        box = {'top': [0, 0, w, n], 'bottom': [0, h-n, w, h],
               'left': [0, 0, n, h], 'right': [w-n, 0, w, h]}[side]
        regions.append({'side': side, 'pixel_box': box, 'rgb': color.tolist(),
                        'normalized_box': [box[0]/w, box[1]/h, box[2]/w, box[3]/h]})
    return regions


def spatial_quality(operation, samples, *, target_box, preview_dir=None, check_edge_residual=False,
                    target_geometry='rectangle', allow_exterior=False):
    """Require full *Agent-declared* target coverage, not hidden target coverage."""
    if target_box is None:
        return {'status': 'failed', 'reason': 'Training-localized target_box is required; pixel change alone is insufficient.'}
    if target_geometry=='exterior' and not allow_exterior:
        return {'status':'failed', 'reason':'Exterior target requires opt-in acceptance.'}
    target = SpatialTransform(type='spatial_mask', region='custom_exterior' if target_geometry=='exterior' else 'custom', custom_box=target_box)
    checks = []
    for index, (reference, source_hash, rgb) in enumerate(samples):
        height, width = rgb.shape[:2]
        mask = _region_mask(operation, width, height)
        claimed = _region_mask(target, width, height)
        changed = _transform_basic(rgb, operation)
        changed_pixels = np.any(rgb != changed, axis=2)
        covered = bool(claimed.any() and not (claimed & ~mask).any())
        protected = bool(np.array_equal(rgb[~mask], changed[~mask]))
        area = float(mask.mean())
        def bounds(region):
            ys, xs = np.where(region)
            return [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if len(xs) else None
        measured_border = np.zeros((height,width),dtype=bool)
        for region in uniform_edge_regions(rgb):
            bx0,by0,bx1,by1=region['pixel_box']
            measured_border[by0:by1,bx0:bx1]=True
        # Area alone is insufficient for a frame: allow a larger mask only when
        # it changes measured constant border pixels and leaves a content region.
        border_exception = bool(allow_exterior and target_geometry=='exterior'
            and operation.region=='custom_exterior' and (~mask).any()
            and not (mask & ~measured_border).any())
        item = {'source_reference': reference, 'source_sha256': source_hash,
                'image_size': [width, height], 'mask_fraction': area,
                'actual_mask_pixel_box': bounds(mask), 'declared_target_pixel_box': bounds(claimed),
                'declared_target_coverage': float((claimed & mask).sum() / max(1, claimed.sum())),
                'uncovered_target_pixels': int((claimed & ~mask).sum()),
                'protected_pixels_identical': protected,
                'changed_pixel_fraction': float(changed_pixels.mean()),
                'passed': covered and protected and 0 < area < 1 and (area <= .5 or border_exception)}
        if allow_exterior:
            item['measured_border_area_exception'] = border_exception
        if check_edge_residual:
            residuals = []
            for region in uniform_edge_regions(rgb):
                x0, y0, x1, y1 = region['pixel_box']
                band = np.zeros((height, width), dtype=bool)
                band[y0:y1, x0:x1] = True
                # Only inspect exact border structures intersecting the Agent target.
                if (band & claimed).any() and (band & ~mask).any():
                    residuals.append({**region, 'uncovered_pixels': int((band & ~mask).sum())})
            item['residual_edge_regions'] = residuals
            item['passed'] = item['passed'] and not residuals
        if preview_dir is not None:
            preview_dir = Path(preview_dir); preview_dir.mkdir(parents=True, exist_ok=True)
            overlay = rgb.copy(); overlay[mask] = (rgb[mask].astype(float) * .5 + [127, 0, 0]).astype('uint8')
            overlay[claimed & ~mask] = [0, 255, 255]
            path = preview_dir / f'{index:03d}.png'
            Image.fromarray(np.concatenate([rgb, overlay, changed], axis=1)).save(path)
            item['preview'] = str(path)
        checks.append(item)
    passed = bool(checks) and all(c['passed'] for c in checks) and any(c['changed_pixel_fraction'] > 0 for c in checks)
    return {'status': 'passed' if passed else 'failed', 'samples': checks,
            'reason': 'Full declared-target coverage and bounded protected-pixel checks required.',
            'scope': 'training_only_no_predictions_no_private_truth',
            'preview_legend': 'original | red intervention mask, cyan uncovered declared target | transformed',
            'limitations': ['Coverage of the Agent-declared box does not certify correct localization or cue removal.',
                           'Task content inside the region may be lost; semantic preservation remains unverified.',
                           'No-op processing controls do not control fill artifacts or semantic damage.']}


def select_spatial_construction(client, *, candidates, observations, descriptor_keys,
                                resources, root, receipt_path, prompt, max_revisions=2,
                                training_diagnostics=None, check_edge_residual=False, allow_exterior=False):
    """Opt-in quality-v3 selection, bounded repairs and tamper-checked resume."""
    if not 0 <= max_revisions <= 2:
        raise ValueError('Construction revision budget must be 0..2')
    samples = training_samples(resources, root)
    payload = {'mode': 'autonomous_visual_intervention', 'construction_protocol': 'spatial-quality-v3.1',
               'frozen_candidates': candidates, 'training_qualitative_observations': observations,
               'available_descriptor_keys': descriptor_keys,
               'training_diagnostics': training_diagnostics or {},
               'training_source_hashes': [[ref, sha] for ref, sha, _ in samples],
               'tool_coordinates': {'corners': 'Each corner preset is the outer 25% of width AND height; bottom_right=[.75,.75,1,1].',
                                    'center': '[.25,.25,.75,.75]', 'border': 'Outer 20% strips',
                                    'custom': 'Normalized x0,y0,x1,y1, rounded at image resolution'},
               'construction_budget': {'max_region_fraction': .5, 'max_revisions': max_revisions},
               'response_schema': SpatialConstructionProposal.model_json_schema()}
    if check_edge_residual:
        payload['construction_protocol'] = 'spatial-quality-v3.2'
        payload['training_exact_border_measurements'] = [
            {'source_reference': ref, 'image_size': [rgb.shape[1], rgb.shape[0]],
             'regions': uniform_edge_regions(rgb)} for ref, _, rgb in samples]
    if allow_exterior:
        payload['construction_protocol']='spatial-quality-v3.3'
        payload['target_geometry_definition']={
            'rectangle':'target_box describes the changed candidate rectangle',
            'exterior':'target_box describes the protected inner rectangle; the candidate target is its exterior. Choose its coordinates from training observations, never presets.',
            'operation':'custom_exterior changes only pixels outside its custom_box; that box remains bit-identical',
            'fill_modes':'constant/mean/noise neutralize; color_complement maps selected RGB to255-RGB, testing reversal only if training evidence supports opposite colors. Compare diagnostic strengths and artifacts before selecting one primary operation.',
            'area_rule':'Ordinary masks remain <=50%; larger exterior masks require every changed location to lie in an exactly measured constant-color border, preserve a nonempty interior, and pass actual visual review.'}
        payload['response_schema']['required'].append('target_geometry')
    identity = digest({'payload': payload, 'prompt': prompt})
    path = Path(receipt_path)
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['input_hash'] != identity or saved['record_hash'] != digest({k:v for k,v in saved.items() if k != 'record_hash'}):
            raise ValueError('Spatial construction resume identity/record changed')
        proposal = SpatialConstructionProposal.model_validate(saved['proposal'])
        localization = TrainingLocalization.model_validate(saved['training_localization'])
        if (saved['selection_input'].get('frozen_training_localization') != localization.model_dump(mode='json')
                or (proposal.candidate_id, proposal.descriptor_key, proposal.target_box, proposal.target_geometry) !=
                   (localization.candidate_id, localization.descriptor_key, localization.target_box, localization.target_geometry)):
            raise ValueError('Spatial construction frozen localization changed')
        if saved['status'] != 'selected' or spatial_quality(proposal.transformation, samples, target_box=proposal.target_box, check_edge_residual=check_edge_residual, target_geometry=proposal.target_geometry, allow_exterior=allow_exterior)['status'] != 'passed':
            raise ValueError('Spatial construction deferred; no training fallback')
        return proposal
    # Localize before exposing operation presets, avoiding preset-size anchoring.
    localization_payload = {k: v for k, v in payload.items()
                            if k not in {'tool_coordinates', 'construction_budget', 'response_schema'}}
    localization_payload.update(mode='autonomous_visual_localization', response_schema=TrainingLocalization.model_json_schema())
    localization_instruction = ''
    if allow_exterior:
        localization_payload['response_schema']['required'].append('target_geometry')
        localization_instruction = (' In this protocol explicitly choose target_geometry. For a frame surrounding content, '
            'exterior means target_box is the protected inner content rectangle, NOT the full-image bounding box. '
            'Use the exact public border pixel measurements to infer its bounds. Rectangle describes an ordinary changed box. '
            'The chosen geometry and box will be immutable in subsequent proposals.')
    localization = TrainingLocalization.model_validate(invoke_agent_client(client, 'quantitative', localization_payload,
        system_prompt=prompt + '\nThis subphase only localizes a full candidate region from training evidence; do not choose an intervention. '
        'Return only TrainingLocalization. Include the complete observed structure, not just its strongest cell or descriptor preset. '
        'A 16x16 grid cell at column c,row r covers x=[c/16,(c+1)/16], y=[r/16,(r+1)/16]. '
        'For example, columns a..b and rows d..e require x0<=a/16,x1>=(b+1)/16,y0<=d/16,y1>=(e+1)/16. '
        'Check that your stated coordinate bounds agree with the training observations and your rationale. No hidden truth, held-out results or model predictions.'+localization_instruction))
    if localization.candidate_id not in {c['candidate_id'] for c in candidates} or localization.descriptor_key not in descriptor_keys:
        raise ValueError('Foreign localization candidate or descriptor')
    payload['frozen_training_localization'] = localization.model_dump(mode='json')
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent/(path.stem+'_localization.json')).write_text(json.dumps({
        'input_hash':identity,'training_localization':localization.model_dump(mode='json'),
        'scope':'training_only_no_heldout_no_model_outcomes_no_truth'},indent=2),encoding='utf-8')
    attempts = []
    instruction = prompt + '\nReturn only the supplied construction proposal schema. First localize the complete candidate target_box from training evidence; it is separate from the operation bounds. Preserve that localization during repairs. Choose an operation covering it, or defer. Do not optimize accuracy, use held-out data, hidden truth or prior validation outcomes. Full geometry coverage is structural acceptance, not proof of semantic preservation or cue erasure.'
    if check_edge_residual:
        instruction += ' Inspect exact public border measurements: coarse grid bounds can omit boundary pixels. Cover the full intersecting constant-color structure with your operation or defer; do not alter the frozen target to pass. These measurements are not hidden truth or semantic certification.'
    if allow_exterior:
        instruction += ' Localization is already complete. Copy candidate_id, descriptor_key, target_geometry and target_box EXACTLY from frozen_training_localization. Choose only the transformation. For exterior, custom_exterior protects its custom_box; do not reinterpret target_box as the outer frame bounds. Compare neutralization and reversal using training support only, never expected validation effects.'
    frozen_target = (localization.candidate_id, localization.descriptor_key, localization.target_box.model_dump(), localization.target_geometry)
    for revision in range(max_revisions + 1):
        proposal = SpatialConstructionProposal.model_validate(invoke_agent_client(client, 'quantitative',
            {**payload, 'revision_index': revision, 'construction_quality_feedback': attempts}, system_prompt=instruction))
        if proposal.candidate_id not in {c['candidate_id'] for c in candidates} or proposal.descriptor_key not in descriptor_keys:
            raise ValueError('Foreign candidate or descriptor')
        if proposal.status == 'deferred':
            quality = {'status': 'failed', 'reason': 'Agent deferred: ' + proposal.rationale}
        else:
            proposed_target = (proposal.candidate_id, proposal.descriptor_key, proposal.target_box.model_dump(), proposal.target_geometry)
            if frozen_target is not None and proposed_target != frozen_target:
                if not allow_exterior:
                    raise ValueError('Construction repair cannot shrink/change the frozen target to pass coverage')
                quality={'status':'failed','reason':'Proposal changed frozen localization; copy frozen candidate, descriptor, geometry and box exactly. Only the operation can be repaired.'}
            else:
                quality = spatial_quality(proposal.transformation, samples, target_box=proposal.target_box,
                                      preview_dir=path.parent / (path.stem + '_previews') / str(revision),
                                      check_edge_residual=check_edge_residual,
                                      target_geometry=proposal.target_geometry, allow_exterior=allow_exterior)
        attempts.append({'revision': revision, 'proposal': proposal.model_dump(mode='json'), 'construction_quality': quality})
        if quality['status'] == 'passed' or proposal.status == 'deferred':
            break
    saved = {'status': 'selected' if quality['status'] == 'passed' else 'deferred_construction_quality',
             'input_hash': identity, 'selection_input': payload, 'proposal': proposal.model_dump(mode='json'),
             'training_localization': localization.model_dump(mode='json'),
             'attempts': attempts, 'selection_owner': 'quantitative_agent',
             'heldout_seen': False, 'truth_seen': False, 'external_ood_used': False}
    saved['record_hash'] = digest(saved)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
    if saved['status'] != 'selected':
        raise ValueError('Spatial construction deferred; no training fallback')
    return proposal
