"""Old policy, receipt schema and pixel/statistical contracts remain executable."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from shortcutminer.protocol.ood_recipe import OODRecipeSelectionPolicy
from shortcutminer.protocol.model_reliance import DatasetShiftSpec
from shortcutminer.experiment.adaptive_screening import paired_drop_interval


def test_plain_v1_has_no_quality_gate_or_new_catalog():
    policy=OODRecipeSelectionPolicy(fields=['patch_corner'])
    assert policy.construction_protocol=='legacy-v1' and policy.catalog_version=='ood-memory-v1'
    from shortcutminer.experiment.ood_recipe_selection_legacy import catalog
    data,_=catalog()
    assert not any(r['recipe_id'].startswith('localized-patch-canonical') for r in data['recipes'])
    assert next(r for r in data['recipes'] if r['recipe_id']=='masked-background-neutralize-v1')['status']=='requires_binding'


def test_existing_explicit_v2_and_new_named_version_keep_quality_gate():
    assert OODRecipeSelectionPolicy(fields=['patch_corner'],max_quality_revisions=2).construction_protocol=='quality-v2'
    assert OODRecipeSelectionPolicy(fields=['patch_corner'],catalog_version='ood-memory-v2').construction_protocol=='quality-v2'
    with pytest.raises(ValueError):
        OODRecipeSelectionPolicy(fields=['patch_corner'],catalog_version='ood-memory-v2',construction_protocol='legacy-v1')


def test_named_quality_catalog_is_separate_from_both_historical_versions():
    from shortcutminer.experiment.ood_recipe_selection_quality import catalog
    explicit=OODRecipeSelectionPolicy(fields=['patch_corner'],catalog_version='ood-memory-v2',construction_protocol='quality-v2')
    data,new_hash=catalog(explicit)
    compat=OODRecipeSelectionPolicy(fields=['patch_corner'],max_quality_revisions=2)
    historical,historical_hash=catalog(compat)
    assert data['catalog_version']=='ood-memory-v2' and historical['catalog_version']=='ood-memory-v1'
    assert data['recipes']==historical['recipes'] and new_hash!=historical_hash


def test_legacy_choice_schema_keeps_original_field_set():
    from shortcutminer.protocol.ood_recipe_legacy import OODRecipeChoice
    assert set(OODRecipeChoice.model_json_schema()['properties'])=={'recipe_id','field','rationale','supporting_tool_call_ids'}
    assert OODRecipeChoice(recipe_id='localized-patch-remove-v1',field='patch_corner',rationale='legacy').model_dump()=={
        'recipe_id':'localized-patch-remove-v1','field':'patch_corner','rationale':'legacy','supporting_tool_call_ids':[]}


def test_legacy_corner_recipe_stays_selectable_without_repair(tmp_path,monkeypatch):
    from test_ood_recipe_selection import patch_fixture,Client
    from shortcutminer.experiment.ood_recipe_selection import select_memory_recipe
    from shortcutminer.experiment import ood_construction_quality
    request,runtime=patch_fixture(tmp_path)
    request.execution_resources=request.execution_resources.model_copy(update={
        'ood_recipe_selection':OODRecipeSelectionPolicy(fields=['color'])})
    monkeypatch.setattr(ood_construction_quality,'construction_quality',lambda *a:pytest.fail('Legacy must not run v2 quality gate'))
    client=Client('localized-patch-remove-v1')
    result=select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert result.execution_resources.patch_ood_specs['color'].mode=='remove' and client.calls==1
    receipt=json.loads((tmp_path/'runs/recipe-test/quantitative/TASK-C001-QT/ood_recipe_selection.json').read_text())
    assert 'attempts' not in receipt and 'construction_quality' not in receipt
    select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert client.calls==1


def test_existing_pair_interval_numbers_are_unchanged():
    labels=['0']*512
    before=['0']*506+['1']*6
    after=['1']*25+['0']*486+['1']
    interval=paired_drop_interval(before,after,labels,alpha=.05/9)
    assert interval['accuracy_drop']==.0390625
    assert interval['lower']==pytest.approx(-.005930379996391462,abs=1e-12)
    assert interval['upper']==pytest.approx(.0826382576857631,abs=1e-12)
    assert interval['method']=='bonferroni_exact_discordance_bounds'


def test_independent_pilot_diagnostics_stays_opt_in():
    assert not DatasetShiftSpec(id_split_values=['id'],ood_split_values=['test']).pilot_diagnostics_when_deferred
