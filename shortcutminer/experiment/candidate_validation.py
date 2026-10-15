"""Opt-in candidate-complete-v1: training-only, Agent-owned measurement plans.

Legacy single-binding views and workflow defaults are deliberately untouched.
"""
from pathlib import Path
from typing import Literal
import csv
import hashlib
import json
import math
import shutil

from pydantic import Field, model_validator

from shortcutminer.experiment.autonomous_audit import public_rows, sha
from shortcutminer.experiment.autonomous_construction import _hash
from shortcutminer.inspection.image import inspect_image_file, flatten_image_descriptors
from shortcutminer.protocol.identify import CandidateId, NonEmptyText, StrictProtocolModel

POLICY = 'candidate-complete-v1'
DESCRIPTORS = ['dimensions', 'intensity', 'rgb_stats', 'saturation', 'entropy', 'edge_density', 'spatial_stats']


class CandidateMeasurement(StrictProtocolModel):
    candidate_id: CandidateId
    status: Literal['measurable', 'deferred', 'not_applicable']
    kind: Literal['descriptor', 'ratio'] | None = None
    descriptor_key: NonEmptyText | None = None
    denominator_key: NonEmptyText | None = None
    rationale: NonEmptyText
    limitations: list[NonEmptyText] = Field(min_length=1)

    @model_validator(mode='after')
    def complete_recipe(self):
        if self.status == 'measurable':
            if self.kind is None or self.descriptor_key is None:
                raise ValueError('Measurable candidate requires one primary measurement')
            if (self.kind == 'ratio') != (self.denominator_key is not None):
                raise ValueError('Only ratios require a denominator')
        elif any(v is not None for v in (self.kind, self.descriptor_key, self.denominator_key)):
            raise ValueError('Deferred/inapplicable candidates cannot bind a measurement')
        return self

    @property
    def measurement_key(self):
        return self.descriptor_key if self.kind == 'descriptor' else f'ratio:{self.descriptor_key}/{self.denominator_key}'


class CandidateMeasurementPlan(StrictProtocolModel):
    policy_version: Literal['candidate-complete-v1']
    measurements: list[CandidateMeasurement] = Field(min_length=1, max_length=16)


def measurements_for_image(path):
    result = inspect_image_file(path, requested_descriptors=DESCRIPTORS,
        max_file_bytes=25000000, max_pixels=20000000, max_frames=1)
    return flatten_image_descriptors(result.descriptors, file_size_bytes=result.byte_size)


def evaluate_measurement(spec, values):
    numerator = values.get(spec.descriptor_key)
    if numerator is None or not math.isfinite(numerator):
        return None
    if spec.kind == 'descriptor':
        return numerator
    denominator = values.get(spec.denominator_key)
    if denominator is None or not math.isfinite(denominator) or denominator == 0:
        return None  # Missing, never silently zero or a guessed epsilon.
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def freeze_candidate_measurements(client, *, root, candidates, observations, receipt_path, prompt):
    root = Path(root).resolve(strict=True); rows = public_rows(root)
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('Measurement planning requires training-only input')
    facts = {}; hashes = {}
    for row in rows:
        path = (root / row['img_filename']).resolve(strict=True); path.relative_to(root)
        hashes[row['img_filename']] = sha(path)
        facts[row['img_id']] = measurements_for_image(path)
    keys = sorted(set.intersection(*(set(v) for v in facts.values())))
    payload = {'run_id': Path(receipt_path).parent.name, 'policy_version': POLICY, 'frozen_candidates': candidates,
        'training_observations': observations, 'available_descriptor_keys': keys,
        'measurement_primitives': ['descriptor', 'ratio'],
        'scope': 'Training-only primary measurement selection, before statistical tests or held-out access. No label, path, split, private truth, arbitrary code or numeric threshold can be a descriptor.',
        'response_schema': CandidateMeasurementPlan.model_json_schema()}
    identity = _hash({'payload': payload, 'source_hashes': hashes,
        'metadata_sha256': sha(root / 'metadata.csv'), 'prompt': prompt})
    expected = {c['candidate_id'] for c in candidates}
    by_candidate = {c['candidate_id']:c for c in candidates}
    def validate(raw):
        plan = CandidateMeasurementPlan.model_validate(raw)
        ids = [s.candidate_id for s in plan.measurements]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError('Plan must cover every frozen candidate exactly once')
        for spec in plan.measurements:
            if spec.status == 'measurable' and by_candidate[spec.candidate_id].get('cue',{}).get('family') == 'metadata':
                raise ValueError('Public intake has only addressing/label/split metadata; do not substitute an image proxy for an unbound metadata hypothesis')
            if spec.status == 'measurable' and not {spec.descriptor_key, *( [spec.denominator_key] if spec.kind == 'ratio' else [])}.issubset(keys):
                raise ValueError('Only registered image descriptors may be measured')
        return plan
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text(encoding='utf-8'))
        if saved['input_hash'] != identity or saved['record_hash'] != _hash({k:v for k,v in saved.items() if k != 'record_hash'}):
            raise ValueError('Frozen measurement plan or training inputs changed')
        plan = validate(saved['plan'])
    else:
        instruction = prompt + '\nCandidateMeasurementPlan JSON only. Cover every candidate. Select ONE primary registered descriptor or ratio before association tests. Explain semantic confounding. Use deferred/not_applicable with a specific reason when no honest measurement exists; never invent a tool or treat unavailable validation as evidence of absence.'
        method = getattr(client, 'invoke_agent_validated', None)
        raw = method('quantitative', payload, system_prompt=instruction, validator=validate) if callable(method) else client.invoke_agent('quantitative', payload, system_prompt=instruction)
        plan = validate(raw)
        resolutions = {}
        for spec in plan.measurements:
            values = [evaluate_measurement(spec, v) for v in facts.values()] if spec.status == 'measurable' else []
            valid = [v for v in values if v is not None]
            reason = ('no_valid_training_measurements' if not valid else
                      'constant_training_measurement' if len(set(valid)) < 2 else 'ready_for_cheap_checks') if spec.status == 'measurable' else spec.status
            resolutions[spec.candidate_id] = {'status': reason, 'valid_count': len(valid),
                'missing_count': len(values) - len(valid), 'rationale': spec.rationale,
                'limitations': spec.limitations, 'absence_of_shortcut_proven': False}
        saved = {'policy_version': POLICY, 'input_hash': identity, 'plan': plan.model_dump(mode='json'),
            'resolutions': resolutions, 'training_source_hashes': hashes,
            'training_metadata_sha256': sha(root / 'metadata.csv'), 'truth_seen': False, 'heldout_seen': False}
        saved['record_hash'] = _hash(saved); _write(receipt_path, saved)
    return plan, saved


def prepare_candidate_complete_view(plan_receipt, constructions, *, discovery_root, training_root, view):
    """Bind all measurable candidates; OOD specs only for accepted constructions."""
    if plan_receipt['record_hash'] != _hash({k:v for k,v in plan_receipt.items() if k != 'record_hash'}):
        raise ValueError('Measurement receipt changed')
    if plan_receipt.get('truth_seen') or plan_receipt.get('heldout_seen'):
        raise ValueError('Plan must be frozen before evaluation access')
    plan = CandidateMeasurementPlan.model_validate(plan_receipt['plan'])
    discovery_root = Path(discovery_root).resolve(strict=True)
    training_root = Path(training_root).resolve(strict=True); view = Path(view).resolve()
    for root in (discovery_root, training_root):
        if view == root or view.is_relative_to(root) or root.is_relative_to(view):
            raise ValueError('Derived view must be separate from original inputs')
    for candidate, receipt in constructions.items():
        if receipt.get('receipt_hash') != _hash({k:v for k,v in receipt.items() if k != 'receipt_hash'}):
            raise ValueError('Construction receipt changed')
        if receipt['construction_metadata_sha256'] != plan_receipt['training_metadata_sha256'] or any(
                plan_receipt['training_source_hashes'].get(ref) != h for ref,h in receipt['training_source_hashes'].items()):
            raise ValueError('Construction and measurement planning used different training inputs')
        spec = next(s for s in plan.measurements if s.candidate_id == candidate)
        if receipt.get('measurement_alignment') is not None:
            from shortcutminer.experiment.intervention_alignment import binding_contract, validate_operation
            check=receipt['measurement_alignment']; expected=binding_contract(spec.model_dump(mode='json'))
            if check.get('binding')!=expected or check.get('status')!='measured_change' or not any(
                    p.get('before') is not None and p.get('after') is not None and p['before']!=p['after']
                    for p in check.get('pairs', [])):
                raise ValueError('Intervention target check does not match the frozen measurement')
            validate_operation(expected,candidate,receipt['proposal']['descriptor_key'],receipt['transformation'])
        if (receipt['status'] != 'selected' or receipt.get('truth_seen') or receipt.get('heldout_seen') or
            receipt['proposal']['candidate_id'] != candidate or receipt['proposal']['descriptor_key'] != spec.measurement_key or
            plan_receipt['resolutions'][candidate]['status'] != 'ready_for_cheap_checks'):
            raise ValueError('Construction does not match the frozen candidate measurement')
    rows = public_rows(training_root); train = public_rows(discovery_root)
    if any(r['split'] != 'train' for r in train) or set(r['split'] for r in rows) != {'train','val','id'}:
        raise ValueError('Authoritative train/val/id partitions required')
    if {r['img_id']:r for r in train} != {r['img_id']:r for r in rows if r['split'] == 'train'}:
        raise ValueError('Training metadata mismatch')
    if sha(discovery_root / 'metadata.csv') != plan_receipt['training_metadata_sha256']:
        raise ValueError('Frozen training metadata changed')
    identity = _hash({'plan': plan_receipt['record_hash'], 'constructions': constructions})
    if view.exists():
        saved = json.loads((view / 'binding_manifest.json').read_text(encoding='utf-8'))
        if saved['input_hash'] != identity or saved['manifest_hash'] != _hash({k:v for k,v in saved.items() if k != 'manifest_hash'}):
            raise ValueError('Frozen complete view identity changed')
        for root, values in [(view, saved['file_hashes']), (training_root, saved['source_hashes'])]:
            for ref, h in values.items():
                path = (root/ref).resolve(strict=True); path.relative_to(root)
                if sha(path) != h: raise ValueError('View or source hash changed')
        return saved
    ready = [s for s in plan.measurements if plan_receipt['resolutions'][s.candidate_id]['status'] == 'ready_for_cheap_checks']
    bindings = {s.candidate_id: 'visual_measurement_' + s.candidate_id.lower() for s in ready}
    prepared = []; hashes = {'metadata.csv': sha(training_root/'metadata.csv')}; groups = {s:set() for s in ('train','val','id')}
    from shortcutminer.inspection.image import load_rgb_image_file
    for row in rows:
        path = (training_root/row['img_filename']).resolve(strict=True); path.relative_to(training_root)
        h = sha(path); hashes[row['img_filename']] = h
        if row['split'] == 'train' and (h != plan_receipt['training_source_hashes'][row['img_filename']] or h != sha(discovery_root/row['img_filename'])):
            raise ValueError('Frozen training pixels changed')
        groups[row['split']].add(hashlib.sha256(load_rgb_image_file(path).rgb.tobytes()).hexdigest())
        values = measurements_for_image(path)
        prepared.append({**row, **{bindings[s.candidate_id]: evaluate_measurement(s, values) for s in ready}})
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','id'),('val','id')]):
        raise ValueError('Exact content overlap between splits')
    view.mkdir(parents=True)
    for row in rows:
        target = view/row['img_filename']; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(training_root/row['img_filename'], target)
    fields = ['img_id','img_filename','y','split', *bindings.values()]
    for name, subset in [('model_metadata.csv', prepared), ('discovery_metadata.csv', [r for r in prepared if r['split']=='train'])]:
        with (view/name).open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(subset)
    saved = {'policy_version': POLICY, 'input_hash': identity, 'visual_measurement_bindings': bindings,
        'candidate_resolutions': plan_receipt['resolutions'], 'structured_specs': {bindings[c]:r['transformation'] for c,r in constructions.items()},
        'truth_read': False, 'proposal_frozen_before_heldout_access': True,
        'source_hashes': hashes, 'split_counts': {s:sum(r['split']==s for r in rows) for s in groups},
        'file_hashes': {p.relative_to(view).as_posix():sha(p) for p in view.rglob('*') if p.is_file()}}
    saved['manifest_hash'] = _hash(saved); _write(view/'binding_manifest.json', saved)
    return saved


def label_group_preparation(label_column):
    """Explicit opt-in config; labels are groups, not shortcut measurements."""
    return {'preparation': {'sampling_strategy': 'group_balanced', 'group_columns': [label_column],
        'max_evidence_records': 40, 'max_contact_sheet_samples': 40},
        'limitations': ['Label groups are task classes, not independent acquisition sources.']}


def write_candidate_dispositions(directory, plan_receipt, construction_outcomes):
    """Explain cheap completion, unavailable measurement and budget deferral separately."""
    directory = Path(directory); summaries = []
    for candidate, resolution in plan_receipt['resolutions'].items():
        results = []
        for file in directory.glob(f'quantitative/TASK-{candidate}-QT/result.json'):
            result = json.loads(file.read_text(encoding='utf-8'))['payload']
            results = [m for m in result.get('measurements', []) if m['experiment_type'] in {'statistical_check','subgroup_evaluation','metric_computation'} and m['execution_outcome']=='completed']
        reason = resolution['status']
        state = 'cheap_checks_completed' if results else ('no_executable_measurement' if reason != 'ready_for_cheap_checks' else 'cheap_checks_not_executed')
        summaries.append({'candidate_id': candidate, 'cheap_state': state, 'measurement_resolution': resolution,
            'cheap_measurement_ids': [m['measurement_id'] for m in results],
            'construction': construction_outcomes.get(candidate, {'status':'not_planned'}),
            'absence_of_shortcut_proven': False})
    _write(directory/'candidate-validation-dispositions.json', {'policy_version':POLICY, 'candidates':summaries,
        'scope':'Execution disposition only; canonical Agent verdicts remain unchanged.'})
    return summaries
