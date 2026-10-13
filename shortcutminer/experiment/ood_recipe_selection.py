"""Versioned dispatch: old configurations keep their original OOD selection."""
from pathlib import Path
from shortcutminer.experiment.ood_recipe_selection_quality import EXPENSIVE, digest, training_facts


def selection_pending(resources):
    policy=resources.ood_recipe_selection
    return bool(policy and set(policy.fields)&set(resources.allowed_scientific_metadata_fields)
                and not any(field in {**resources.color_ood_specs,**resources.patch_ood_specs,**resources.background_ood_specs}
                            for field in policy.fields if field in resources.allowed_scientific_metadata_fields))


def select_memory_recipe(request, client, runtime, *, prompt, runs_root):
    policy=request.execution_resources.ood_recipe_selection
    if policy.construction_protocol=='legacy-v1':
        from shortcutminer.experiment.ood_recipe_selection_legacy import select_memory_recipe as select
        from shortcutminer.experiment.agent_workspace import default_agent_workspace_dir
        legacy_memory=(default_agent_workspace_dir('quantitative')/'ood-selection-legacy-v1.md').read_text(encoding='utf-8')
        prompt += '\n\nThis invocation uses the frozen legacy-v1 OOD selection protocol. The following legacy selection memory takes precedence over quality-v2 instructions for this subphase.\n'+legacy_memory
    else:
        from shortcutminer.experiment.ood_recipe_selection_quality import select_memory_recipe as select
    return select(request,client,runtime,prompt=prompt,runs_root=runs_root)
