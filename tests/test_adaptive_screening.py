import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from shortcutminer.experiment.adaptive_screening import run_probe, paired_drop_interval
from shortcutminer.experiment.evidence_promotion import evaluate_promotion, load_or_evaluate_promotion
from shortcutminer.protocol.model_reliance import AdaptivePairedScreeningOptions
from shortcutminer.protocol.promotion import PromotionPolicy

pytestmark = pytest.mark.filterwarnings('ignore:The argument .*Tensor.* is deprecated:DeprecationWarning')


def options(**updates):
    return AdaptivePairedScreeningOptions(minimum_effect=.02, family_alpha=.05,
        candidate_budget=3, pair_checkpoints=[128,256,512], probe_fraction=.5,
        sensitivity_effects=[.01,.02,.05], **updates)


@pytest.mark.parametrize('kind,expected', [('weak','stop_low_effect'), ('strong','expand_models'), ('uncertain','deferred_pair_budget_exhausted')])
def test_real_paired_counts_control_budget_and_simultaneous_alpha(kind, expected):
    labels = ['0']*441
    calls=[]
    def predict(n):
        calls.append(n)
        after = ['0']*n
        if kind == 'strong': after = ['1']*n
        if kind == 'uncertain': after[:max(1,round(.02*n))] = ['1']*max(1,round(.02*n))
        return labels[:n], after, labels[:n]
    report = run_probe(options(), labels=labels, sample_ids=list(map(str,range(441))), predict=predict)
    assert report['decision']==expected
    assert all(look['alpha_at_look']==pytest.approx(.05/9) for look in report['looks'])
    assert len(calls)<=3 and max(calls)<=441
    assert all(set(look['sensitivity'])=={'0.01','0.02','0.05'} for look in report['looks'])
    if kind=='strong': assert calls==[128]
    if kind=='uncertain': assert calls==[128,256,441]


def test_bad_controls_and_unlearned_task_do_not_become_negative_evidence():
    report=run_probe(options(),labels=['0']*256,sample_ids=list(map(str,range(256))),
        predict=lambda n:(['0']*n,['0']*n,['1']*n))
    assert report['decision']=='deferred_processing_control_failed'
    report=run_probe(options(),labels=['0']*256,sample_ids=list(map(str,range(256))),
        predict=lambda n:pytest.fail('Unlearned task must not be interpreted'),task_learned=False)
    assert report['decision']=='deferred_task_not_learned'


def test_interval_accounts_for_both_lost_and_gained_predictions():
    a=['0']*128+['1']*128;b=['1']*128+['0']*128
    interval=paired_drop_interval(a,b,['0']*256,alpha=.005)
    assert interval['accuracy_drop']==0 and interval['lower']<0<interval['upper']


def test_zero_signal_receives_bounded_probe_and_unmaterializable_candidates_do_not(tmp_path):
    policy=PromotionPolicy(policy_version='probe-test',calibration_status='test_only',
        strategy='budgeted_paired_probe',max_probe_candidates=1)
    inputs={'candidate_ids':['C001','C002','C003'], 'qualitative_coverage':[], 'measurements':[
        {'candidate_id':'C001','measurement_id':'M1','experiment_type':'statistical_check',
         'execution_outcome':'completed','metrics':{'effect_size':0}},
        {'candidate_id':'C002','measurement_id':'M2','experiment_type':'statistical_check',
         'execution_outcome':'completed','metrics':{'effect_size':.1}}]}
    result=evaluate_promotion('probe',policy,inputs,probe_candidate_ids=['C001'])
    assert [r.decision for r in result.results]==['promoted','deferred_insufficient_evidence','deferred_insufficient_evidence']
    cached=load_or_evaluate_promotion(tmp_path,'probe',policy,inputs,probe_candidate_ids=['C001'])
    assert cached==result
    with pytest.raises(ValueError,match='inputs/policy'):
        load_or_evaluate_promotion(tmp_path,'probe',policy,inputs,probe_candidate_ids=['C002'])


def test_final_evaluation_excludes_reserved_probe_rows(tmp_path):
    from shortcutminer.experiment.model_reliance import _filter_evaluation_rows
    from shortcutminer.protocol.experiment import TaskModelEvaluationParameters
    frame=pd.DataFrame({'id':['train','probe','final'],'path':['a.png']*3,'label':['0']*3,'split':['train','id','id']})
    (tmp_path/'rows.csv').write_text(frame.to_csv(index=False))
    (tmp_path/'model.json').write_text(json.dumps({'training':{'train_sample_ids':['train'],'test_sample_ids':['probe','final'],'adaptive_probe_sample_ids':['probe']}}))
    access=SimpleNamespace(dataset_file=lambda p:tmp_path/p,run_file=lambda p:tmp_path/p,
                           load_csv=lambda p:pd.read_csv(tmp_path/p))
    parameters=TaskModelEvaluationParameters(model_reference='model.json',metadata_file='rows.csv',
        sample_id_column='id',sample_path_column='path',label_column='label',split_column='split',
        split_values=['id'],evaluation_partition='configured_split')
    rows=_filter_evaluation_rows(parameters,access,artifact=object())
    assert rows['id'].tolist()==['final']


def test_shared_probe_weights_stop_and_resume_without_training_again(tmp_path, monkeypatch):
    """Actual tensor/NPZ path on CPU; no provider or real dataset experiment."""
    import torch
    from shortcutminer.task_models import torchvision_matrix as module
    from shortcutminer.protocol.model_reliance import TorchvisionTrainingOptions
    from PIL import Image
    monkeypatch.setattr(module,'torch_runtime',lambda config:torch)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    monkeypatch.setattr(module,'initialization',lambda architecture:(None,{'fixture_only':True}))
    monkeypatch.setattr(module.TorchvisionResNetMatrix,'_build',lambda self,architecture:
        torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,2)))
    config=TorchvisionTrainingOptions(epochs=1,minimum_epochs=1,workers=0,amp=False,checkpoint_interval=1)
    paths=[]
    for i in range(8):
        path=tmp_path/f'{i}.png';Image.new('RGB',(8,8),(255,0,0) if i%2 else (0,0,255)).save(path);paths.append(path)
    cache=tmp_path/'shared'; seen=[]
    for task in ['first','second']:
        model=module.TorchvisionResNetMatrix(task,config)
        model.fit_matrix(paths,['0','1']*4,paths,['0','1']*4,tmp_path/task,task,
            probe_callback=lambda fitted,profile:seen.append(profile['weights_sha256']) or False,shared_cache=cache)
        assert len(model.profiles)==1
    assert len(seen)==2 and seen[0]==seen[1]
    assert model.profiles[0]['shared_current_run_reuse'] is True


def test_new_color_config_is_valid_and_legacy_config_is_unchanged():
    from pathlib import Path
    from shortcutminer.protocol.audit import AuditExecutionOptions
    root=Path(__file__).resolve().parents[1]
    new=json.loads((root/'configs/experiments/colorized-mnist-shortcutminer-adaptive-v2.json').read_text(encoding='utf-8'))
    parsed=AuditExecutionOptions.model_validate({**new['execution_options'],'max_train_samples':6359,'max_samples':10000})
    assert parsed.adaptive_screening.minimum_effect==.02
    old=json.loads((root/'configs/experiments/colorized-mnist-shortcutminer-formal.json').read_text(encoding='utf-8'))
    assert old['promotion_policy']['metric_rules'][0]['threshold']==.5


@pytest.mark.parametrize('signal,recipe_id', [('weak',None),('strong',None),('unlearned',None),
    ('strong','foreground-red-green-swap-v1'),('strong','foreground-color-remove-v1'),
    ('strong','localized-patch-canonical-remove-v2'),('strong','localized-patch-canonical-randomize-v2'),('strong','autonomous-spatial')])
def test_five_agent_adaptive_path_with_cpu_tensor_fixture(tmp_path, monkeypatch, signal, recipe_id):
    """Production graph and actual NPZ training; tiny CPU fixture, no live APIs."""
    import torch
    from PIL import Image
    from pathlib import Path
    from shortcutminer.task_models import torchvision_matrix as module
    from shortcutminer.audit import build_audit_run_request
    from shortcutminer.protocol.audit import AuditRequest
    from shortcutminer.runner import run, build_local_tool_runtimes
    from test_core_image_audit import ImageAuditClient, CHECKS
    torch.set_num_threads(2)
    monkeypatch.setattr(module,'torch_runtime',lambda config:torch)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU test fixture')
    monkeypatch.setattr(module,'initialization',lambda architecture:(None,{'fixture_only':True}))
    class TinyFixture(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,images):
            means=torch.tensor([.485,.456,.406]).view(1,3,1,1)
            stds=torch.tensor([.229,.224,.225]).view(1,3,1,1)
            rgb=images*stds+means
            value=rgb.min(1).values.mean((1,2))-.5 if signal=='weak' else (rgb[:,0]-rgb[:,1 if recipe_id and recipe_id!='autonomous-spatial' else 2]).mean((1,2))
            if recipe_id and recipe_id.startswith('localized-patch-'):
                extent=rgb.shape[-1]//4
                value=rgb[:,0,-extent:,-extent:].mean((1,2))-rgb[:,0,:extent,:extent].mean((1,2))
            if recipe_id=='autonomous-spatial':
                extent=rgb.shape[-1]//4
                value=(rgb[:,0,-extent:,-extent:]-rgb[:,2,-extent:,-extent:]).mean((1,2))
            if signal=='unlearned': value=rgb.mean((1,2,3))*0
            return torch.stack([-value,value],dim=1)*self.scale
    monkeypatch.setattr(module.TorchvisionResNetMatrix,'_build',lambda self,architecture:TinyFixture())
    root=tmp_path/'data';root.mkdir();rows=[]
    for i in range(1056):
        split='train' if i<16 else 'val' if i<24 else 'id' if i<1048 else 'test'
        label=i%2; site=label if signal=='strong' else (i//2)%2
        # Labels depend on invariant grayscale content; palette can be independent.
        intensity=np.full((8,8,1),230 if label else 20,dtype=np.uint8)
        intensity[0,0,0]=i%256;intensity[0,1,0]=i//256
        rgb=np.maximum(intensity,
                       np.array([255,0,0] if site else [0,0,255],dtype=np.uint8))
        if recipe_id and recipe_id!='autonomous-spatial' and not recipe_id.startswith('localized-patch-'):
            rgb=np.zeros((8,8,3),dtype=np.uint8)
            rgb[:,:,0 if site else 1]=intensity[:,:,0]
        elif recipe_id and recipe_id!='autonomous-spatial':
            rgb=np.full((8,8,3),230 if label else 20,dtype=np.uint8)
            rgb[3,3]=i%256;rgb[3,4]=i//256
            start=6 if label else 0
            rgb[start:start+2,start:start+2]=[255,0,0]
        path=f'{i}.png';Image.fromarray(rgb).save(root/path)
        rows.append({'id':str(i),'path':path,'label':label,'split':split,'site':site})
    pd.DataFrame(rows).to_csv(root/'model.csv',index=False)
    pd.DataFrame(rows[:16]).to_csv(root/'discovery.csv',index=False)
    opts=dict(sample_id_column='id',sample_path_column='path',split_column='split',model_metadata_file='model.csv',
        train_split_values=['train'],validation_split_values=['val'],test_split_values=['id','test'],evaluation_split_values=['id'],
        model_backend='torchvision_resnet_matrix',feature_version='torchvision_rgb224_v1',max_train_samples=16,max_samples=1024,
        training_options={'epochs':1,'minimum_epochs':1,'workers':0,'amp':False,'checkpoint_interval':1,'augmentation':'digit_preserving'},
        color_ood_specs={'site':{'type':'global_color','mode':'cycle_rgb'}},adaptive_screening=options().model_dump())
    opts['ood_specs']={'site':{'id_split_values':['id'],'ood_split_values':['test']}}
    if signal=='weak':
        opts['ood_specs']['site']['pilot_diagnostics_when_deferred']=True
    if recipe_id=='autonomous-spatial':
        opts.pop('color_ood_specs')
        opts['visual_ood_specs']={'site':{'type':'spatial_mask','region':'bottom_right','fill':'constant','fill_color':[0,0,255]}}
    elif recipe_id:
        opts.pop('color_ood_specs')
        opts['ood_recipe_selection']={'catalog_version':'ood-memory-v1','fields':['site']}
        if recipe_id.startswith('localized-patch-canonical-'):
            opts['ood_recipe_selection']['construction_protocol']='quality-v2'
    request=build_audit_run_request(AuditRequest(dataset_path=str(root),metadata_file='discovery.csv',
        label_information={'source':'metadata_column','column':'label'},execution_options=opts),run_id='adaptive-fixture',runs_root=tmp_path/'runs')
    request=request.model_copy(update={'executive_config':{'audit_options':{'required_quantitative_checks':CHECKS,'max_samples_per_task':1024}},
        'metadata':{**request.metadata,'promotion_policy':{'policy_version':'test-probe','calibration_status':'test_only',
        'strategy':'budgeted_paired_probe','max_probe_candidates':3}}})
    class MemoryClient(ImageAuditClient):
        def invoke_agent(self,name,payload,*,system_prompt=None):
            if payload.get('mode') == 'ood_recipe_selection':
                assert name=='quantitative' and 'ood-memory-v1' in system_prompt
                assert not list((Path(request.runs_root)/request.run_id).glob('models/task/*/profiles.json'))
                self.called.append(name)
                assert recipe_id in {r['recipe_id'] for r in payload['eligible_recipes']}
                return {'recipe_id':recipe_id,'field':'site','rationale':'Training-only cue and pixel facts support this controlled color intervention.',
                        'supporting_tool_call_ids':[r['tool_call_id'] for r in payload['successful_current_task_cheap_results']]}
            return super().invoke_agent(name,payload,system_prompt=system_prompt)
    client=MemoryClient() if recipe_id and recipe_id!='autonomous-spatial' else ImageAuditClient()
    runtimes=build_local_tool_runtimes(run_id=request.run_id,dataset_root=root,runs_root=request.runs_root,
        max_tool_calls_per_run=80,max_tool_calls_per_task=30)
    result=run(request,agent_client=client,tool_runtimes=runtimes)
    assert result.status=='COMPLETED'
    assert set(client.called)=={'identify','executive','quantitative','qualitative','summarize'}
    directory=Path(request.runs_root)/request.run_id
    if recipe_id and recipe_id!='autonomous-spatial':
        selected=json.loads(next(directory.glob('quantitative/*/ood_recipe_selection.json')).read_text())
        assert selected['choice']['recipe_id']==recipe_id
        assert selected['selection_owner']=='quantitative_agent'
        assert selected['selection_input']['input_facts']['training_images_checked']==16
    profiles=json.loads(next(directory.glob('models/task/*/profiles.json')).read_text())
    report=profiles['adaptive_report']
    if signal=='strong':
        assert report['decision']=='expand_models' and len(profiles['profiles'])==6
        manifest=json.loads(next(directory.glob('interventions/**/intervention_manifest.json')).read_text())
        final_ids={p['original_sample_id'] for p in manifest['pairs']}
        assert len(final_ids)==512 and not final_ids.intersection(report['reserved_probe_ids'])
        independent=json.loads(next(directory.glob('models/ood/TASK-C001-QT/*.json')).read_text())
        assert independent['comparison_mode']=='independent' and len(independent['profiles'])==6
        assert all(p['id_count']==512 and p['ood_count']==8 for p in independent['profiles'])
    elif signal=='weak':
        assert report['decision']=='stop_low_effect' and len(profiles['profiles'])==1
        assert not list(directory.glob('interventions/**/intervention_manifest.json'))
        diagnostic=json.loads(next(directory.glob('models/ood/**/*.json')).read_text())
        assert diagnostic['diagnostic_only'] and not diagnostic['final_confirmation_executed']
        assert len(diagnostic['profiles'])==1 and diagnostic['profiles'][0]['id_count']==512
    else:
        assert report['decision']=='deferred_task_not_learned' and len(profiles['profiles'])==1
        assert not list(directory.glob('interventions/**/intervention_manifest.json'))
    assert result.workflow_state['promotion_result']['results'][0]['decision']=='promoted'
    assert (directory/'summarize/result.json').is_file()
    # The ToolRegistry's isolated staging must also preserve cache handoff.
    from shortcutminer.tooling.protocol import ToolCall
    original_training=json.loads(next(directory.glob('experiments/**/task_model_training-*.json')).read_text())
    second=runtimes['quantitative'].fork_for_task(candidate_id='C002',task_id='TASK-C002-QT',
        runtime_constraints=[],requested_checks=['task_model_training'],explicitly_authorized_artifacts=[])
    parameters={**original_training['parameters'],'model_id':'second-fixture-model'}
    reused=second.execute(ToolCall(tool_call_id='shared-probe-fixture',tool_name='run_experiment',agent_name='quantitative',
        run_id=request.run_id,candidate_id='C002',task_id='TASK-C002-QT',experiment_type='task_model_training',
        arguments={'candidate_id':'C002','task_id':'TASK-C002-QT','experiment_type':'task_model_training','parameters':parameters}))
    assert reused.status=='success', reused.error
    reused_profiles=json.loads((directory/'models/task/second-fixture-model/profiles.json').read_text())
    assert reused_profiles['profiles'][0]['shared_current_run_reuse'] is True
