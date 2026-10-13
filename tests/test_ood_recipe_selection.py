"""Memory selection cannot inspect held-out pixels or bypass eligibility."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from shortcutminer.experiment.ood_recipe_selection import select_memory_recipe, training_facts
from shortcutminer.experiment.quantitative_plan import QuantitativePlanUnmaterializable
from shortcutminer.protocol.quantitative import QuantitativeExecutionResources
from shortcutminer.protocol.audit import AuditExecutionOptions


def fixture(tmp_path):
    root=tmp_path/'data'; root.mkdir()
    rows=[]
    for i in range(4):
        rgb=np.zeros((8,8,3),dtype=np.uint8);rgb[2:6,2:6,i%2]=100+i
        Image.fromarray(rgb).save(root/f'{i}.png')
        rows.append(dict(id=str(i),path=f'{i}.png',y=str(i%2),color=['red','green'][i%2],split='train'))
    # Missing held-out file is intentional: planning must never open it.
    rows.append(dict(id='heldout',path='must-not-open.png',y='0',color='red',split='test'))
    pd.DataFrame(rows).to_csv(root/'model.csv',index=False)
    resources=QuantitativeExecutionResources(dataset_name='fixture',metadata_file='model.csv',
        available_metadata_fields=['id','path','y','color','split'],label_column='y',
        allowed_scientific_metadata_fields=['color'],sample_id_column='id',sample_path_column='path',split_column='split',
        model_metadata_file='model.csv',train_split_values=['train'],
        ood_recipe_selection={'fields':['color']})
    candidate=SimpleNamespace(candidate_id='C001',model_dump=lambda **kw: {'candidate_id':'C001','hypothesis':'color cue'})
    task=SimpleNamespace(task_id='TASK-C001-QT',model_dump=lambda **kw: {'task_id':'TASK-C001-QT'})
    request=SimpleNamespace(run_id='recipe-test',candidate=candidate,task=task,execution_resources=resources)
    request.model_copy=lambda update: SimpleNamespace(**{**request.__dict__,**update})
    runtime=SimpleNamespace(context=SimpleNamespace(dataset_root=root),history=[])
    return request,runtime


class Client:
    def __init__(self,recipe='foreground-red-green-swap-v1',field='color',evidence=None):
        self.recipe=recipe; self.field=field; self.evidence=evidence or []; self.calls=0
    def invoke_agent(self,name,payload,**kwargs):
        self.calls+=1
        assert name=='quantitative' and payload['mode']=='ood_recipe_selection'
        assert payload['input_facts']['training_images_checked']==4
        return dict(recipe_id=self.recipe,field=self.field,rationale='Choose the scoped controlled manipulation.',
                    supporting_tool_call_ids=self.evidence)


@pytest.mark.parametrize('recipe',['foreground-red-green-swap-v1','foreground-color-remove-v1'])
def test_choice_is_provider_owned_and_resume_reuses_record(tmp_path,recipe):
    request,runtime=fixture(tmp_path);client=Client(recipe)
    selected=select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert selected.execution_resources.color_ood_specs['color'].mode==('swap_red_green' if recipe.endswith('swap-v1') else 'grayscale_max')
    again=select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert again.execution_resources==selected.execution_resources and client.calls==1
    path=tmp_path/'runs/recipe-test/quantitative/TASK-C001-QT/ood_recipe_selection.json'
    data=json.loads(path.read_text());data['choice']['rationale']='tampered'
    path.write_text(json.dumps(data))
    with pytest.raises(QuantitativePlanUnmaterializable,match='record_changed'):
        select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')


@pytest.mark.parametrize('client',[Client('localized-patch-remove-v1'),Client(field='y'),Client(evidence=['foreign-task'])])
def test_reject_unsupported_control_field_or_foreign_evidence_before_training(tmp_path,client):
    request,runtime=fixture(tmp_path)
    with pytest.raises(QuantitativePlanUnmaterializable):
        select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert not list((tmp_path/'runs').glob('**/model.json'))


def test_no_eligible_recipe_has_no_provider_or_fallback(tmp_path):
    request,runtime=fixture(tmp_path)
    for i in range(4): Image.new('RGB',(8,8),(10,20,30)).save(runtime.context.dataset_root/f'{i}.png')
    client=Client()
    with pytest.raises(QuantitativePlanUnmaterializable,match='no_eligible'):
        select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert client.calls==0
    receipt=tmp_path/'runs/recipe-test/quantitative/TASK-C001-QT/ood_recipe_selection.json'
    assert json.loads(receipt.read_text())['status']=='unavailable'


def test_training_pixel_change_invalidates_resume(tmp_path):
    request,runtime=fixture(tmp_path);client=Client()
    select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    Image.new('RGB',(8,8),(250,0,0)).save(runtime.context.dataset_root/'0.png')
    with pytest.raises(QuantitativePlanUnmaterializable,match='identity_changed'):
        select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')


def test_native_config_has_no_preselected_color_recipe():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/experiments/colored-mnist-irm-shortcutminer-memory-v2.json').read_text(encoding='utf-8-sig'))
    options=AuditExecutionOptions.model_validate({**config['execution_options'],'max_train_samples':39280,'max_samples':10900})
    assert options.ood_recipe_selection and not options.color_ood_specs
    assert options.adaptive_screening.minimum_effect==.02
    with pytest.raises(ValueError,match='preselected'):
        AuditExecutionOptions.model_validate({**options.model_dump(mode='json'),
            'color_ood_specs':{'color':{'type':'global_color','mode':'swap_red_green'}}})


def patch_fixture(tmp_path):
    request,runtime=fixture(tmp_path)
    from shortcutminer.protocol.ood_recipe import OODRecipeSelectionPolicy
    request.execution_resources=request.execution_resources.model_copy(update={
        'ood_recipe_selection':OODRecipeSelectionPolicy(fields=['color'],construction_protocol='quality-v2')})
    from shortcutminer.intervention.corner_patch import boxes
    for i in range(4):
        rgb=np.random.default_rng(i).integers(0,240,(32,32,3),dtype=np.uint8)
        x0,y0,x1,y1=boxes(rgb,4)[i];rgb[y0:y1,x0:x1]=[255,0,0]
        Image.fromarray(rgb).save(runtime.context.dataset_root/f'{i}.png')
    return request,runtime


def test_agent_repairs_legacy_source_location_trace_before_training(tmp_path):
    request,runtime=patch_fixture(tmp_path)
    class RepairClient(Client):
        def invoke_agent(self,name,payload,**kwargs):
            if self.calls:
                failure=payload['construction_quality_feedback'][-1]
                assert failure['construction_quality']['status']=='failed'
                assert 'location' in failure['construction_quality']['reason']
                self.recipe='localized-patch-canonical-randomize-v2'
            return super().invoke_agent(name,payload,**kwargs)
    client=RepairClient('localized-patch-remove-v1')
    selected=select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert client.calls==2 and selected.execution_resources.patch_ood_specs['color'].mode=='canonical_randomize_position'
    record=json.loads((tmp_path/'runs/recipe-test/quantitative/TASK-C001-QT/ood_recipe_selection.json').read_text())
    assert len(record['attempts'])==2 and record['construction_quality']['status']=='passed'
    assert all(item['passed'] for item in record['construction_quality']['samples'])
    select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert client.calls==2


def test_failed_quality_has_no_forced_recipe_or_training(tmp_path):
    request,runtime=patch_fixture(tmp_path);client=Client('localized-patch-remove-v1')
    with pytest.raises(QuantitativePlanUnmaterializable,match='quality_not_approved'):
        select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert client.calls==3 and not list((tmp_path/'runs').glob('**/model.json'))
    record=json.loads((tmp_path/'runs/recipe-test/quantitative/TASK-C001-QT/ood_recipe_selection.json').read_text())
    assert record['status']=='deferred_construction_quality'


def test_agent_can_defer_instead_of_repairing(tmp_path):
    request,runtime=patch_fixture(tmp_path);client=Client('unavailable')
    with pytest.raises(QuantitativePlanUnmaterializable,match='quality_not_approved'):
        select_memory_recipe(request,client,runtime,prompt='policy',runs_root=tmp_path/'runs')
    assert client.calls==1


def test_agent_binds_authorized_small_models_and_foreground_prompt(tmp_path,monkeypatch):
    request,runtime=fixture(tmp_path)
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/waterbirds-shortcutminer-memory-v2.json').read_text(encoding='utf-8'))
    from shortcutminer.protocol.ood_recipe import OODRecipeSelectionPolicy
    policy=OODRecipeSelectionPolicy.model_validate({**config['execution_options']['ood_recipe_selection'],'fields':['color']})
    request.execution_resources=request.execution_resources.model_copy(update={'ood_recipe_selection':policy})
    from shortcutminer.intervention import local_foreground
    bound=[]
    class Provider:
        def __init__(self,spec): bound.append(spec); self.last_details={'coverage':.25,'oracle_mask_used':False}
        def segment(self,**kwargs):
            mask=np.zeros((8,8),dtype=bool);mask[2:6,2:6]=True
            return SimpleNamespace(mask=mask)
        def close(self): pass
    monkeypatch.setattr(local_foreground,'LocalForegroundSegmentationProvider',Provider)
    class BackgroundClient(Client):
        def invoke_agent(self,name,payload,**kwargs):
            response=super().invoke_agent(name,payload,**kwargs)
            assert 'local-dino-sam-vitmatte' in payload['available_local_foreground_tools']
            return {**response,'local_foreground_tool':'local-dino-sam-vitmatte','object_prompt':'foreground animal.'}
    selected=select_memory_recipe(request,BackgroundClient('masked-background-neutralize-v1'),runtime,prompt='policy',runs_root=tmp_path/'runs')
    operation=selected.execution_resources.background_ood_specs['color']
    assert operation.local_foreground.object_prompt=='foreground animal.' and bound[0]==operation.local_foreground
    assert operation.local_foreground.sam.files_sha256==policy.local_foreground_tools['local-dino-sam-vitmatte'].sam.files_sha256
    assert not selected.execution_resources.color_ood_specs
