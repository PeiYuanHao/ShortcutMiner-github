"""Public retrieval, temporal isolation and post-run evidence-gated method learning."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from shortcutminer.experiment.knowledge_research import ResearchSession,PaperSearchInput,digest
from shortcutminer.experiment.learned_shortcuts import freeze_learned_snapshot,learn_after_completion,certification,sha
from shortcutminer.experiment.identify_executor import execute_identify
from shortcutminer.tooling.builtins import build_default_tool_registry,EmptyInput
from shortcutminer.tooling.base import ToolExecutionContext
from shortcutminer.tooling.runtime import ToolRuntime
from shortcutminer.protocol.quantitative import QuantitativeMeasurement
from shortcutminer.task_models.reference import _canonical_hash
from test_identify_executor import _create_test_dataset,_create_request,_valid_draft


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def test_search_budget_cache_privacy_and_failure_are_explicit(tmp_path):
    calls=[]
    def transport(query,limit,timeout):calls.append(query);return [{'title':'Primary paper','url':'https://arxiv.org/abs/2004.07780'}]
    policy={'enabled':True,'max_queries':1}
    session=ResearchSession(policy,tmp_path/'research',forbidden_terms=['privatecase'],transport=transport)
    assert session.search(PaperSearchInput(keywords=['privatecase exposure'])).status=='rejected_private_query'
    request=PaperSearchInput(keywords=['spurious correlation','exposure'])
    a=session.search(request);assert a.status=='completed' and len(calls)==1
    assert session.search(request)==a and len(calls)==1
    resumed=ResearchSession(policy,tmp_path/'research',transport=transport)
    assert resumed.search(request)==a
    assert resumed.search(PaperSearchInput(keywords=['new mechanism'])).status=='budget_exhausted'
    disabled=ResearchSession({},tmp_path/'disabled',transport=lambda *a:pytest.fail('offline contacted network'))
    assert disabled.search(request).status=='disabled'
    failed=ResearchSession({'enabled':True},tmp_path/'failed',transport=lambda *a: (_ for _ in ()).throw(TimeoutError('private transport details')))
    result=failed.search(request)
    assert result.status=='unavailable' and 'private transport details' not in json.dumps(result.model_dump())


@pytest.mark.parametrize('topic',['E:/data/images','y equals 0','ratio 90 percent','api key','https://private.com','hidden truth'])
def test_queries_cannot_export_records_or_credentials(topic):
    with pytest.raises(ValidationError):PaperSearchInput(keywords=[topic])


def test_identify_actually_selects_and_executes_a_literature_query(tmp_path,monkeypatch):
    _create_test_dataset(tmp_path)
    request=_create_request(tmp_path)
    calls=[]
    monkeypatch.setattr('shortcutminer.experiment.knowledge_research.fetch_arxiv',lambda *a:calls.append(a) or [{'title':'Source style','url':'https://arxiv.org/abs/2004.07780'}])
    class Client:
        count=0
        def invoke_agent(self,agent,payload,**kwargs):
            assert agent=='identify'
            self.count+=1
            if self.count==1:return {'tool_calls':[{'name':'search_shortcut_literature','arguments':{'keywords':['source style','spurious correlation']}}]}
            assert payload['tool_runtime']['results'][0]['output']['status']=='completed'
            return _valid_draft()
    runtime=ToolRuntime(build_default_tool_registry(),run_id=request.run_id,agent_name='identify',dataset_root=tmp_path,runs_root=tmp_path/'runs')
    report=execute_identify(request,42,Client(),runs_root=tmp_path/'runs',tool_runtime=runtime,research_policy={'enabled':True,'include_learned':True})
    assert report.candidates and len(calls)==1
    assert (tmp_path/'runs'/request.run_id/'identify/learned-memory.json').exists()
    assert len(list((tmp_path/'runs'/request.run_id/'identify/research').glob('*.json')))==1
    runtime.context=__import__('dataclasses').replace(runtime.context,runtime_constraints=frozenset({'offline'}))
    with pytest.raises(Exception,match='offline constraints'):execute_identify(request,42,Client(),runs_root=tmp_path/'runs',tool_runtime=runtime,research_policy={'enabled':True})


def fixture_evidence(runs_root):
    root=runs_root/'complete-run';root.mkdir(parents=True)
    write(root/'run.json',{'status':'COMPLETED'})
    options={'architectures':['modela','modelb'],'seeds':[1,2]}
    profiles=[];predictions={};intervals=[];member_profiles=[]
    for architecture in options['architectures']:
        for seed in options['seeds']:
            name=f'{architecture}-seed{seed}'
            weights='models/frozen/'+name+'.npz';path=root/weights;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
            profiles.append({'architecture':architecture,'seed':seed,'weights_reference':weights,'weights_sha256':sha(path)})
            predictions[name]=[{'sample_id':str(i),'label':'0','prediction_original':'0','prediction_counterfactual':'1'} for i in range(10)]
            intervals.append({'metric':name+'_accuracy_drop','available':True,'method':'bootstrap','lower':.5,'upper':1.0})
            member_profiles.append({'profile':name,'accuracy_drop':1.0,'pair_count':10})
    full={'options':options,'profiles':profiles,'adaptive_report':{'decision':'expand_models','reserved_probe_ids':['reserved-pilot']}}
    write(root/'models/frozen/profiles.json',full)
    model_hash=_canonical_hash(full)
    write(root/'models/frozen/model.json',{'run_id':'complete-run','backend':'torchvision_resnet_matrix','backend_version':'torchvision_resnet_matrix_v1','model_hash':model_hash,'weights_reference':'models/frozen/profiles.json','profiles_sha256':sha(root/'models/frozen/profiles.json')})
    paired={'run_id':'complete-run','candidate_id':'C001','task_id':'TASK-C001-QT','model_hash':model_hash,'members':{'comparison_mode':'paired','model_hash':model_hash,'profiles':member_profiles,'predictions':predictions,'uncertainty':intervals}}
    write(root/'interventions/frozen/measurement.json',paired)
    metrics={'accuracy_delta':-1.0}
    measurement=QuantitativeMeasurement(measurement_id='M-real',run_id='complete-run',candidate_id='C001',task_id='TASK-C001-QT',plan_step_id='confirmation',planning_check='ood_evaluation',experiment_type='ood_evaluation',tool_call_id='TC-confirmation',backend_identity='torchvision_resnet_matrix_v1',execution_outcome='completed',metrics=metrics,artifact_references=['interventions/frozen/measurement.json','models/frozen/model.json'],design_strength='interventional')
    write(root/'tools/quantitative/TC-confirmation.json',{'tool_call_id':'TC-confirmation','tool_name':'run_experiment','agent_name':'quantitative','run_id':'complete-run','candidate_id':'C001','task_id':'TASK-C001-QT','plan_step_id':'confirmation','planning_check':'ood_evaluation','experiment_type':'ood_evaluation','status':'success','output':{'outcome':'completed','metrics':metrics,'summary':'Frozen paired model measurement','artifact_references':measurement.artifact_references,'metadata':{'backend':'torchvision_resnet_matrix_v1'}},'started_at':'start','completed_at':'end','duration_ms':1.0})
    candidate={'candidate_id':'C001','cue':{'family':'appearance','name':'Observed nuisance cue'},'description':'Observed association','observation':'Training appearance differs','hypothesis':'Model may use nuisance cue','evidence_sources':['image_statistics'],'priority':'high','status':'unverified'}
    identify={'protocol_version':'1.0','run_id':'complete-run','message_id':'identify','source':'identify','target':'executive','message_type':'candidate_report','in_reply_to':'request','dataset_summary':{'dataset':'privatecase','samples_inspected':10,'metadata_used':False},'candidates':[candidate],'limitations':[],'artifact_references':[]}
    summary={'protocol_version':'1.0','run_id':'complete-run','message_id':'summary','source':'summarize','target':'orchestrator','message_type':'summary_report','in_reply_to':'executive','payload':{'overview':'Scoped support','findings':[{'candidate_id':'C001','assessment':'supported','summary':'Scoped model support','quantitative_result_ids':['quantitative'],'qualitative_result_ids':[],'limitations':['Class semantics remain uncertain'],'recommendations':[]}],'limitations':[],'artifact_references':[]}}
    write(root/'summarize/result.json',summary)
    quant={'protocol_version':'1.0','run_id':'complete-run','message_id':'quantitative','source':'quantitative','target':'summarize','message_type':'quantitative_analysis_result','in_reply_to':'task','payload':{'task_id':'TASK-C001-QT','candidate_id':'C001','hypothesis':'Model may use nuisance cue','status':'completed','analysis_type':'paired model validation','metrics':metrics,'measurements':[measurement.model_dump(mode='json')],'evidence_level':'Q4','evidence_direction':'supports','evidence_summary':'Scoped dependence','limitations':['Instance independence assumed'],'artifact_references':measurement.artifact_references,'recommended_followup':[]}}
    state={'summarize_result':summary,'identify_result':identify,'quantitative_result':{'results':[quant]},'dataset_info':{'name':'privatecase','root':'public-input'},'metadata':{'audit_request':{'execution_options':{'adaptive_screening':{'minimum_effect':.02}}}}}
    return state,measurement,root


class AbstractClient:
    def invoke_agent(self,agent,payload,**kwargs):
        assert agent=='identify' and payload['mode']=='learned_shortcut_abstraction'
        return {'mechanism_id':'novel-acquisition-signature','mechanism':'Acquisition style nuisance','signature':'Source processing style is predictive beyond task content','validation_methods':['Agent-selected paired style neutralization with frozen models'], 'limitations':['Scoped to approved interventions; labels and instance independence need checks']}


def test_verified_method_is_written_and_only_future_snapshots_receive_it(tmp_path):
    state,measurement,root=fixture_evidence(tmp_path)
    before=freeze_learned_snapshot(tmp_path,'before-learning');assert not before['entries']
    learned=learn_after_completion(state,AbstractClient(),tmp_path,'complete-run',{'enabled':True})
    assert learned['entries'][0]['status']=='learned'
    assert freeze_learned_snapshot(tmp_path,'before-learning')==before
    future=freeze_learned_snapshot(tmp_path,'future-run');assert len(future['entries'])==1
    public=json.dumps(future)
    assert 'privatecase' not in public and 'complete-run' not in public and 'minimum_effect' not in public
    assert learn_after_completion(state,AbstractClient(),tmp_path,'complete-run',{'enabled':True})==learned
    context=ToolExecutionContext(run_id='future-run',agent_name='identify',learned_memory_snapshot=future)
    memory=build_default_tool_registry().resolve('inspect_shortcut_methods').execute(EmptyInput(),context)
    assert memory.catalog['learned_method_memory']==future


@pytest.mark.parametrize('failure',['no_models','probe_overlap','uncertain','nonfinite_interval','weights_changed','tool_mismatch'])
def test_ineligible_backend_evidence_never_becomes_identify_memory(tmp_path,failure):
    state,measurement,root=fixture_evidence(tmp_path)
    path=root/'interventions/frozen/measurement.json';paired=json.loads(path.read_text())
    if failure=='no_models':paired.pop('members')
    if failure=='probe_overlap':
        for rows in paired['members']['predictions'].values():rows[0]['sample_id']='reserved-pilot'
    if failure=='uncertain':paired['members']['uncertainty'][0]['lower']=-.01
    if failure=='nonfinite_interval':paired['members']['uncertainty'][0]['lower']=float('nan')
    if failure=='weights_changed':(root/'models/frozen/modela-seed1.npz').write_bytes(b'tampered')
    if failure=='tool_mismatch':
        tool=root/'tools/quantitative/TC-confirmation.json';value=json.loads(tool.read_text());value['candidate_id']='C002';write(tool,value)
    write(path,paired)
    result=learn_after_completion(state,AbstractClient(),tmp_path,'complete-run',{'enabled':True})
    assert result['entries'][0]['status']=='deferred'
    assert not list((tmp_path/'knowledge/mechanisms').glob('*.json'))


def test_noncompleted_or_unfrozen_policy_cannot_learn(tmp_path):
    state,measurement,root=fixture_evidence(tmp_path)
    state['metadata']={}
    assert learn_after_completion(state,AbstractClient(),tmp_path,'complete-run',{'enabled':True})['status']=='deferred'
    (root/'knowledge/learning-result.json').unlink()
    write(root/'run.json',{'status':'FAILED'})
    with pytest.raises(ValueError,match='durable run completion'):learn_after_completion(state,AbstractClient(),tmp_path,'complete-run',{'enabled':True})
