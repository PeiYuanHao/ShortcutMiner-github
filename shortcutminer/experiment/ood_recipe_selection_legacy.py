"""Quantitative chooses an eligible memory recipe before any costly execution."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from shortcutminer.agent_client import invoke_agent_client
from shortcutminer.experiment.agent_workspace import default_agent_workspace_dir
from shortcutminer.experiment.quantitative_plan import QuantitativePlanUnmaterializable
from shortcutminer.intervention.protocol import GlobalColorTransform, CornerPatchTransform
from shortcutminer.protocol.ood_recipe_legacy import OODRecipeChoice
from shortcutminer.storage.stage_artifacts import save_stage_json
from shortcutminer.inspection.image import load_rgb_image_file


OPERATIONS = {
    'foreground-red-green-swap-v1': 'swap_red_green',
    'foreground-color-remove-v1': 'grayscale_max',
    'saturated-background-channel-cycle-v1': 'cycle_rgb',
    'localized-patch-remove-v1': 'remove',
    'localized-patch-randomize-v1': 'randomize_position',
}
EXPENSIVE = {'task_model_training', 'task_model_evaluation', 'ood_evaluation', 'counterfactual_intervention'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def selection_pending(resources):
    policy = resources.ood_recipe_selection
    return bool(policy and set(policy.fields) & set(resources.allowed_scientific_metadata_fields)
                and not any(field in {**resources.color_ood_specs, **resources.patch_ood_specs} for field in policy.fields
                            if field in resources.allowed_scientific_metadata_fields))


def catalog():
    path = default_agent_workspace_dir('quantitative') / 'ood_recipes.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['catalog_version'] != 'ood-memory-v1':
        raise QuantitativePlanUnmaterializable('unknown_ood_memory_version')
    for entry in data['recipes']:
        if entry['status'] == 'executable':
            if entry['recipe_id'] not in OPERATIONS or entry['operation'] != {
                'type': 'corner_patch' if entry['recipe_id'].startswith('localized-patch-') else 'global_color', 'mode': OPERATIONS[entry['recipe_id']]
            }:
                raise QuantitativePlanUnmaterializable('ood_memory_operation_not_allowlisted')
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


def _file(root, reference):
    path = (root / reference).resolve(strict=True)
    path.relative_to(root)
    return path


def training_facts(resources, root):
    """Check input compatibility only; never inspect validation, ID or OOD pixels."""
    root = Path(root).resolve(strict=True)
    if not all([resources.model_metadata_file, resources.split_column, resources.sample_path_column,
                resources.train_split_values]):
        raise QuantitativePlanUnmaterializable('ood_memory_training_inputs_missing')
    frame = pd.read_csv(_file(root, resources.model_metadata_file), dtype=str, keep_default_na=False)
    train = frame[frame[resources.split_column].isin(resources.train_split_values)]
    if train.empty or len(train) > resources.max_rows:
        raise QuantitativePlanUnmaterializable('ood_memory_training_input_budget_invalid')
    canonical, saturated = True, True
    patch_patterns = None
    pixel_identity = hashlib.sha256()
    for reference in train[resources.sample_path_column]:
        path = _file(root, reference)
        rgb = load_rgb_image_file(path).rgb
        pixel_identity.update(str(reference).encode()); pixel_identity.update(hashlib.sha256(path.read_bytes()).digest())
        canonical &= bool(not rgb[:, :, 2].any() and rgb.max() > 0
                          and not np.any((rgb[:, :, 0] > 0) & (rgb[:, :, 1] > 0)))
        channels = [c for c in range(3) if np.all(rgb[:, :, c] == 255)]
        saturated &= any(np.array_equal(rgb[:, :, (c+1)%3], rgb[:, :, (c+2)%3])
                         for c in channels)
        from shortcutminer.intervention.corner_patch import boxes
        patterns={}
        for size in range(2,min(8,(min(rgb.shape[:2])-1)//2)+1):
            for x0,y0,x1,y1 in boxes(rgb,size):
                block=rgb[y0:y1,x0:x1]; color=tuple(int(v) for v in block[0,0])
                if np.all(block==color): patterns[(size,color)]=patterns.get((size,color),0)+1
        unique={key for key,count in patterns.items() if count==1}
        patch_patterns=unique if patch_patterns is None else patch_patterns & unique
    patch_binding=None
    if patch_patterns:
        largest=max(size for size,color in patch_patterns)
        matches=[(size,color) for size,color in patch_patterns if size==largest]
        if len(matches)==1:
            size,color=matches[0];patch_binding={'patch_size':size,'patch_color':list(color)}
    return {'training_images_checked': len(train), 'training_pixels_sha256': pixel_identity.hexdigest(),
            'canonical_red_green_foreground': canonical, 'saturated_palette_with_grayscale_content': bool(saturated),
            'verified_unique_constant_corner_patch': patch_binding,
            'scope': 'training_pixels_only_no_heldout_statistics_or_expected_verdict'}


def select_memory_recipe(request, client, runtime, *, prompt, runs_root):
    resources = request.execution_resources
    policy = resources.ood_recipe_selection
    fields = sorted(set(policy.fields) & set(resources.allowed_scientific_metadata_fields))
    data, catalog_hash = catalog()
    facts = training_facts(resources, runtime.context.dataset_root)
    eligible = []
    for item in data['recipes']:
        if item['status'] != 'executable':
            continue
        mode = OPERATIONS[item['recipe_id']]
        suitable = (bool(facts['verified_unique_constant_corner_patch']) if item['recipe_id'].startswith('localized-patch-')
                    else facts['saturated_palette_with_grayscale_content'] if mode == 'cycle_rgb'
                    else facts['canonical_red_green_foreground'])
        if suitable:
            eligible.append(item)
    history = [item.model_dump(mode='json') for item in runtime.history
               if item.status == 'success' and item.run_id == request.run_id
               and item.candidate_id == request.candidate.candidate_id and item.task_id == request.task.task_id
               and item.tool_name == 'run_experiment' and item.experiment_type not in EXPENSIVE]
    payload = {'mode': 'ood_recipe_selection', 'run_id': request.run_id,
               'candidate': request.candidate.model_dump(mode='json'), 'task': request.task.model_dump(mode='json'),
               'authorized_fields': fields, 'input_facts': facts, 'eligible_recipes': eligible,
               'unavailable_recipes': [item for item in data['recipes'] if item not in eligible],
               'successful_current_task_cheap_results': history, 'response_schema': OODRecipeChoice.model_json_schema()}
    identity = digest(payload)
    receipt_path = (Path(runs_root) / request.run_id / 'quantitative' / request.task.task_id
                    / 'ood_recipe_selection.json') if runs_root is not None else None

    def validate(raw):
        choice = OODRecipeChoice.model_validate(raw)
        if choice.field not in fields or choice.recipe_id not in {item['recipe_id'] for item in eligible}:
            raise QuantitativePlanUnmaterializable('ood_memory_choice_not_eligible')
        if not set(choice.supporting_tool_call_ids).issubset({item['tool_call_id'] for item in history}):
            raise QuantitativePlanUnmaterializable('ood_memory_choice_foreign_evidence')
        return choice

    if not eligible:
        if runs_root is not None:
            save_stage_json(runs_root, run_id=request.run_id, stage='quantitative', scope=request.task.task_id,
                            filename='ood_recipe_selection.json', payload={'status': 'unavailable', 'selection_input': payload,
                            'input_hash': identity, 'catalog_sha256': catalog_hash})
        raise QuantitativePlanUnmaterializable('no_eligible_memory_ood_recipe_no_fallback')
    if receipt_path is not None and receipt_path.exists():
        saved = json.loads(receipt_path.read_text(encoding='utf-8'))
        if saved.get('input_hash') != identity or saved.get('catalog_sha256') != catalog_hash:
            raise QuantitativePlanUnmaterializable('ood_memory_resume_identity_changed')
        choice = validate(saved['choice'])
        if saved.get('record_hash') != digest({key: value for key, value in saved.items() if key != 'record_hash'}):
            raise QuantitativePlanUnmaterializable('ood_memory_resume_record_changed')
    else:
        selection_prompt = prompt + '\n\nOOD recipe selection subphase takes precedence over final-result instructions. ' \
            'Return only OODRecipeChoice. Select exactly one eligible recipe based on this candidate, training input facts ' \
            'and successful cheap evidence. Do not choose using held-out outcomes, invent parameters, claim a measurement, ' \
            'or output a QuantitativeAgentDraft/tool call. If no suitable recipe, return recipe_id="unavailable" ' \
            'with a reason; execution will stop without a fallback.'
        raw = invoke_agent_client(client, 'quantitative', payload, system_prompt=selection_prompt)
        choice = validate(raw)
        bound = (CornerPatchTransform(type='corner_patch',mode=OPERATIONS[choice.recipe_id],
                 **facts['verified_unique_constant_corner_patch'],seed=resources.seed).model_dump(mode='json')
                 if choice.recipe_id.startswith('localized-patch-') else {'type':'global_color','mode':OPERATIONS[choice.recipe_id]})
        saved = {'status': 'selected', 'selection_owner': 'quantitative_agent', 'selection_input': payload,
                 'input_hash': identity, 'catalog_sha256': catalog_hash, 'choice': choice.model_dump(mode='json'),
                 'transformation': bound}
        saved['record_hash'] = digest(saved)
        if runs_root is not None:
            save_stage_json(runs_root, run_id=request.run_id, stage='quantitative', scope=request.task.task_id,
                            filename='ood_recipe_selection.json', payload=saved)
    if choice.recipe_id.startswith('localized-patch-'):
        operation=CornerPatchTransform(type='corner_patch',mode=OPERATIONS[choice.recipe_id],
                    **facts['verified_unique_constant_corner_patch'],seed=resources.seed)
        updated=resources.model_copy(update={'patch_ood_specs':{**resources.patch_ood_specs,choice.field:operation}})
    else:
        operation = GlobalColorTransform(type='global_color', mode=OPERATIONS[choice.recipe_id])
        updated = resources.model_copy(update={'color_ood_specs': {**resources.color_ood_specs, choice.field: operation}})
    return request.model_copy(update={'execution_resources': updated})
