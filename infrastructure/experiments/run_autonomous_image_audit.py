"""Opt-in generic two-phase image audit; credentials come only from .env.formal.

No dataset names, planting parameters, hidden truth or OOD recipes are preset here.
Existing experiment entry points remain unchanged.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shortcutminer.audit import build_audit_run_request, persist_audit_request
from shortcutminer.experiment.autonomous_audit import public_rows, prepare_autonomous_view, sha
from shortcutminer.experiment.autonomous_construction import select_autonomous_construction
from shortcutminer.experiment.loader import load_model_runtime_config
from shortcutminer.experiment.waterbirds_paper import source_identity, archive_source
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.protocol.model import ModelRuntimeConfigSet
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runner import run, resume_run, build_local_tool_runtimes
from shortcutminer.runtime.formal_environment import formal_credentials
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient, UrllibStreamingJsonTransport
from shortcutminer.runtime.qualitative_multimodal import QualitativeMultimodalClient
from shortcutminer.inspection.semantic_provider import OpenAICompatibleSemanticVisualInspector
from shortcutminer.tooling.runtime import ToolBudget


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


class FrozenIdentifyReplay:
    def __init__(self,draft):self.draft=draft
    def invoke_agent(self,name,payload,**kwargs):
        if name!='identify':raise ValueError('Replay is restricted to frozen Identify draft')
        return deepcopy(self.draft)


def execute(request,text_client,vision_client,root,runs_root,vlm_config,identify=None):
    directory=runs_root/request.run_id
    payload=request.model_dump(mode='json');saved=directory/'frozen_execution_config.json'
    resume=(directory/'run.json').exists()
    if saved.exists():
        if json.loads(saved.read_text(encoding='utf-8'))!=payload:raise ValueError('Frozen execution request changed')
        original=json.loads((directory/'source_identity.json').read_text(encoding='utf-8'))
        current=source_identity(require_clean=False)
        if original['source']['source_tree_sha256']!=current['source_tree_sha256']:
            raise ValueError('Source snapshot changed; use a new run ID')
        if sha(directory/'source_snapshot.zip')!=original['archive_sha256']:
            raise ValueError('Frozen source archive changed')
    else:
        if resume:raise ValueError('Existing run has no authoritative frozen config')
        persist_audit_request(request)
        source=source_identity(require_clean=False)
        write(directory/'source_identity.json',{'source':source,'archive_sha256':archive_source(directory/'source_snapshot.zip',source['source_tree_sha256'])})
        write(saved,payload)
    if resume and json.loads((directory/'run.json').read_text())['status']=='COMPLETED':return directory
    tools=build_local_tool_runtimes(run_id=request.run_id,dataset_root=root,runs_root=runs_root,restore_persisted_task_results=resume,max_tool_calls_per_run=150,max_tool_calls_per_task=40)
    for runtime in tools.values():
        options=request.metadata.get('audit_request',{}).get('execution_options') or {}
        selected_limit=max(500,int(options.get('max_samples') or 500))
        runtime.budget=ToolBudget(max_tool_calls=40,timeout_seconds=300,max_selected_samples=selected_limit,
            experiment_timeout_seconds={'task_model_training':None,'task_model_evaluation':1800,'ood_evaluation':1800})
    clients=dict(agent_client=text_client,identify_client=identify or text_client,
        qualitative_client=QualitativeMultimodalClient(vision_client,root,max_images=40,contact_sheet=True,runs_root=runs_root),tool_runtimes=tools,
        semantic_inspector=OpenAICompatibleSemanticVisualInspector(vision_client))
    if resume:resume_run(request.run_id,runs_root=runs_root,**clients)
    else:run(request,**clients)
    if json.loads((directory/'run.json').read_text())['status']!='COMPLETED':raise RuntimeError('Audit did not complete; inspect saved run diagnostics')
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Public training-only discovery directory')
    parser.add_argument('--training-input',type=Path,required=True,help='Public train/val/id directory, opened after construction acceptance')
    parser.add_argument('--profile',type=Path,required=True,help='Frozen training, early-stop and promotion protocol JSON; dataset-specific OOD fields are discarded')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--task',required=True,help='Public task description; do not supply hidden mechanisms')
    parser.add_argument('--vlm-max-tokens',type=int,default=8192,help='Frozen visual protocol output budget, matching prior complete audit entries')
    parser.add_argument('--isolated-discovery',action='store_true',help='Exclude learned memory from discovery; run in a new runs-root to isolate prior artifacts')
    parser.add_argument('--runs-root',type=Path,default=ROOT/'runs')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--proposal-only',action='store_true')
    mode.add_argument('--validation',action='store_true')
    args=parser.parse_args()
    if not 2048 <= args.vlm_max_tokens <= 32768:
        parser.error('--vlm-max-tokens must be between 2048 and 32768')
    # RunRequest validates run identity before writing any artifact directory.
    profile=json.loads(args.profile.read_text(encoding='utf-8'))
    knowledge=json.loads((ROOT/'configs/knowledge/research-policy-v1.json').read_text(encoding='utf-8'))
    if args.isolated_discovery:
        knowledge['research']['include_learned']=False
    public=public_rows(args.input)
    if any(r['split']!='train' for r in public):raise ValueError('Discovery input contains non-training rows')
    with formal_credentials():
        text=load_model_runtime_config(ROOT/profile['text_model_config'])
        vlm=load_model_runtime_config(ROOT/profile['vlm_model_config']).model_copy(update={'max_tokens':args.vlm_max_tokens})
        models=ModelRuntimeConfigSet(shared=text,per_agent={'qualitative':vlm})
        client=OpenAICompatibleClient(text,runs_root=args.runs_root)
        vision=OpenAICompatibleClient(vlm,runs_root=args.runs_root,transport=UrllibStreamingJsonTransport())
        discovery=build_audit_run_request(AuditRequest(dataset_path=str(args.input),metadata_file='metadata.csv',label_information={'source':'metadata_column','column':'y'},dataset_description=args.task),runs_root=args.runs_root,run_id=args.run_id+'-discovery',model_runtime=models)
        value=discovery.model_dump(mode='json')
        value['identify_config']={'budget':{'max_candidates':5,'max_samples':40,'max_dataset_entries':20000,'max_metadata_rows':10000},'visual_discovery':{'enabled':True,'sample_id_column':'img_id','sample_path_column':'img_filename','split_column':'split','selected_split':'train','max_samples':40,'semantic_enabled':True}}
        discovery=RunRequest.model_validate(value)
        value=discovery.model_dump(mode='json')
        value['identify_config']['research_policy']=knowledge['research']
        discovery=RunRequest.model_validate(value)
        initial=execute(discovery,client,vision,args.input,args.runs_root,vlm)
        report=json.loads((initial/'identify/output.json').read_text(encoding='utf-8'))
        visual=json.loads((initial/'identify/visual_discovery.json').read_text(encoding='utf-8'))
        keys=sorted({k for s in visual['sample_observations'] for k in (s.get('descriptors') or {}).get('values',{})})
        observations=[]
        for path in sorted((initial/'qualitative').glob('*/result.json')):
            observations.append(json.loads(path.read_text(encoding='utf-8'))['payload'])
        resources=SimpleNamespace(model_metadata_file='metadata.csv',split_column='split',train_split_values=['train'],sample_path_column='img_filename',max_train_samples=len(public))
        control=args.runs_root/(args.run_id+'-construction')
        receipt=select_autonomous_construction(client,vision,run_id=args.run_id+'-construction',candidates=report['candidates'],observations=observations,descriptor_keys=keys,resources=resources,root=args.input,receipt_path=control/'construction.json',task_description=args.task,training_diagnostics={'group_summaries':visual['group_summaries'],'descriptor_definitions':visual.get('descriptor_definitions',{})})
        print(json.dumps({'construction':receipt['status'],'receipt':str(control/'construction.json')}),flush=True)
        if receipt['status']!='selected' or args.proposal_only:return
        view=args.runs_root/(args.run_id+'-input')
        binding=prepare_autonomous_view(receipt,args.input,args.training_input,view)
        field=binding['feature_field'];candidate=receipt['proposal']['candidate_id'];operation=receipt['transformation']
        options={k:v for k,v in profile['execution_options'].items() if k not in {'ood_specs','ood_recipe_selection','background_ood_specs','color_ood_specs','patch_ood_specs','visual_ood_specs','visual_measurement_bindings','excluded_scientific_fields'}}
        counts=binding['split_counts']
        options.update(model_metadata_file='model_metadata.csv',sample_id_column='img_id',sample_path_column='img_filename',split_column='split',train_split_values=['train'],validation_split_values=['val'],test_split_values=['id'],evaluation_split_values=['id'],max_train_samples=counts['train'],max_samples=counts['id'],max_intervention_samples=counts['id'],association_variable_mode='numeric_categorical',visual_measurement_bindings={candidate:field})
        options['color_ood_specs' if operation['type']=='global_color' else 'visual_ood_specs']={field:operation}
        request=build_audit_run_request(AuditRequest(dataset_path=str(view),metadata_file='discovery_metadata.csv',label_information={'source':'metadata_column','column':'y'},execution_options=options,intake_budget={'max_entries':20000},dataset_description=args.task+'; Frozen training-only discovery and Agent-selected paired diagnostic. Semantic preservation and source-instance independence unverified.'),runs_root=args.runs_root,run_id=args.run_id,model_runtime=models)
        value=request.model_dump(mode='json')
        value['identify_config']=deepcopy(discovery.identify_config)
        value['identify_config']['visual_discovery']['semantic_enabled']=False
        value['identify_config']['research_policy']={**knowledge['research'],'enabled':False,'include_learned':False}
        value['executive_config']={'audit_options':{'validation_depth':'deep','max_samples_per_task':counts['id'],'qualitative_metadata_policy':'candidate-bound-v2','required_quantitative_checks':['association_test','task_model_training','task_model_evaluation','ood_evaluation']}}
        value['metadata'].update(knowledge_policy=knowledge['learning'],promotion_policy=profile['promotion_policy'],source_policy='user_authorized_current_source_snapshot',construction_receipt_hash=receipt['receipt_hash'],discovery_report_sha256=sha(initial/'identify/output.json'),data_view_input_hash=sha(view/'binding_manifest.json'),paper_protocol={'confirmatory_blind_evidence':False,'note':'Development execution; a confirmatory claim requires untouched evaluation and a pre-frozen protocol.','isolated_discovery':args.isolated_discovery,'initial_discovery_replayed':True,'intervention_owner':'quantitative_agent','actual_paired_visual_review':True,'hidden_truth_used':False})
        request=RunRequest.model_validate(value)
        raw=json.loads((initial/'identify/raw_output.json').read_text(encoding='utf-8'))
        directory=execute(request,client,vision,view,args.runs_root,vlm,FrozenIdentifyReplay(raw))
        print(json.dumps({'run_id':request.run_id,'directory':str(directory),'status':'COMPLETED'}),flush=True)


if __name__=='__main__':main()
