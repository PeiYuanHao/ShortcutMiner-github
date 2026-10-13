"""Actual five-Agent Waterbirds workflow: fresh quantitative CUDA training and OOD.

Prepare immutable data views first. Live credentials are loaded only in --live.
Never imports earlier standalone baseline checkpoints or its prediction tables.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shortcutminer.experiment.waterbirds_paper import inspect_dataset, sha, identity, write, source_identity, archive_source
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.audit import build_audit_run_request, persist_audit_request
from shortcutminer.experiment.loader import load_model_runtime_config
from shortcutminer.protocol.model import ModelRuntimeConfigSet
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runner import run, resume_run, build_local_tool_runtimes
from shortcutminer.runtime.formal_environment import formal_credentials
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.runtime.qualitative_multimodal import QualitativeMultimodalClient
from shortcutminer.inspection.semantic_provider import OpenAICompatibleSemanticVisualInspector
from shortcutminer.tooling.runtime import ToolBudget

ROOT=Path(__file__).resolve().parents[1]


def prepare(config):
    """Fixed split protocol independent of training seeds; all originals unchanged."""
    from sklearn.model_selection import train_test_split
    rows,data=inspect_dataset({'dataset_root':config['dataset_root'],'metadata_file':'metadata.csv',
        'expected_metadata_sha256':config['expected_metadata_sha256']},decode=True)
    root=Path(config['derived_dataset_root']).resolve(); root.mkdir(parents=True,exist_ok=True)
    fingerprint=identity({'data_sha256':data['data_sha256'],'split_protocol':config['split_protocol']})
    manifest_path=root/'input_manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text())
        if existing['input_hash']!=fingerprint: raise RuntimeError('Existing data view identity changed')
        for reference,expected in existing['view_hashes'].items():
            if sha(root/reference)!=expected: raise RuntimeError('Frozen data view changed')
        for r in data['image_manifest']:
            if sha(root/r['img_filename'])!=r['sha256']: raise RuntimeError('Derived image bytes changed')
        return root,existing
    official_train=[r for r in rows if r['split']=='0']
    _,id_positions=train_test_split(list(range(len(official_train))),test_size=config['split_protocol']['id_fraction'],
        random_state=config['split_protocol']['seed'],stratify=[r['y']+':'+r['place'] for r in official_train])
    id_set={official_train[i]['img_id'] for i in id_positions}
    model_rows=[]; partition_map=[]
    source=Path(config['dataset_root']).resolve(strict=True)
    for row in rows:
        split=('id' if row['img_id'] in id_set else 'train') if row['split']=='0' else {'1':'val','2':'ood'}[row['split']]
        model_rows.append({k:(split if k=='split' else row[k]) for k in ('img_id','img_filename','y','split','place')})
        partition_map.append({'img_id':row['img_id'],'original_split':row['split'],'execution_split':split})
        origin=(source/row['img_filename']).resolve(strict=True); origin.relative_to(source)
        target=(root/row['img_filename']).resolve(); target.relative_to(root); target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=sha(origin): raise RuntimeError('Derived image identity changed')
        else: os.link(origin,target)
    columns=['img_id','img_filename','y','split','place']
    for name,selected in [('model_metadata.csv',model_rows),('discovery_metadata.csv',[r for r in model_rows if r['split']=='train'])]:
        with (root/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(selected)
    write(root/'partition_map.json',partition_map)
    counts={s:sum(r['split']==s for r in model_rows) for s in ('train','id','val','ood')}
    manifest={'input_hash':fingerprint,'data_sha256':data['data_sha256'],'partition_counts':counts,
        'view_hashes':{f:sha(root/f) for f in ('model_metadata.csv','discovery_metadata.csv','partition_map.json')},
        'split_protocol':config['split_protocol'],'source_dataset':str(source),'original_metadata_sha256':config['expected_metadata_sha256'],
        'image_strategy':'hard links to verified original image bytes; no baseline predictions or checkpoints',
        'discovery_mode':'metadata_assisted_train_partition_only'}
    write(manifest_path,manifest)
    return root,manifest


def execute(config_path,run_id,*,live):
    config=json.loads(Path(config_path).read_text(encoding='utf-8')); root,data=prepare(config)
    if not live:
        print(json.dumps({'status':'PREPARED','dataset_root':str(root),**data},indent=2)); return
    counts=data['partition_counts']
    options={**config['execution_options'],'max_train_samples':counts['train'],'max_samples':counts['ood']}
    audit=AuditRequest(dataset_path=str(root),metadata_file='discovery_metadata.csv',
        label_information={'source':'metadata_column','column':'y'},
        dataset_description=('Metadata-assisted dataset audit. Discovery and Stage-1 see only the frozen training partition. '
            'ID is an independently held-out subset of the original training distribution. '
            + ('OOD is constructed from the same held-out ID images by the caller-frozen local foreground/background intervention; '
               'predicted segmentation and label preservation require independent review. ' if options.get('background_ood_specs') else
               'OOD is the official test partition evaluated independently. ')
            + 'Hypotheses are unverified; no expected conclusion is supplied.'),
        execution_options=options,intake_budget={'max_entries':30000})
    with formal_credentials():
        text=load_model_runtime_config(config['text_model_config']); vlm=load_model_runtime_config(config['vlm_model_config'])
        vlm=vlm.model_copy(update={'max_tokens':8192})
        runtimes_config=ModelRuntimeConfigSet(shared=text,per_agent={'qualitative':vlm})
        request=build_audit_run_request(audit,runs_root=str(ROOT/'runs'),run_id=run_id,model_runtime=runtimes_config)
        request=RunRequest.model_validate({**request.model_dump(mode='json'),
            'identify_config':{'budget':{'max_candidates':5,'max_samples':8,'max_dataset_entries':30000,'max_metadata_rows':100000},
                'visual_discovery':{'enabled':True,'sample_id_column':'img_id','sample_path_column':'img_filename',
                    'split_column':'split','selected_split':'train','max_samples':8,'semantic_enabled':False}},
            'executive_config':{'audit_options':{'validation_depth':'deep','max_samples_per_task':counts['ood'],
                'required_quantitative_checks':['association_test','task_model_training','task_model_evaluation','ood_evaluation']}},
            'qualitative_config':{'preparation':{'max_evidence_records':8,'max_contact_sheet_samples':8,
                'sampling_strategy':'group_balanced','group_columns':['y']}},
            'metadata':{**request.metadata,'promotion_policy':config['promotion_policy'],'paper_protocol':config['paper_protocol'],
                'data_view_input_hash':data['input_hash'],'source_policy':'user_authorized_current_source_snapshot'}})
        persist_audit_request(request)
        directory=ROOT/'runs'/run_id
        source=source_identity(require_clean=False)
        write(directory/'core_execution_receipt.json',{'source':source,'source_archive_sha256':archive_source(directory/'source_snapshot.zip',source['source_tree_sha256']),
            'config_sha256':sha(Path(config_path)),'data_view':data,'paper_sha256':sha(Path(config['paper_protocol']['source_pdf'])),
            'scope':'actual_five_agent_workflow_not_standalone_baseline','training_from_baseline_checkpoints':False})
        tools=build_local_tool_runtimes(run_id=run_id,dataset_root=root,runs_root=ROOT/'runs',max_tool_calls_per_run=150,max_tool_calls_per_task=40)
        for agent in tools:
            tools[agent].budget=ToolBudget(max_tool_calls=40,timeout_seconds=300,
                experiment_timeout_seconds={'task_model_training':None,'task_model_evaluation':1800,'ood_evaluation':1800},max_selected_samples=counts['ood'])
        text_client=OpenAICompatibleClient(text,runs_root=ROOT/'runs')
        vlm_client=OpenAICompatibleClient(vlm,runs_root=ROOT/'runs')
        result=run(request,agent_client=text_client,qualitative_client=QualitativeMultimodalClient(vlm_client,root),
            tool_runtimes=tools,semantic_inspector=OpenAICompatibleSemanticVisualInspector(vlm_client))
        print(json.dumps({'run_id':run_id,'status':json.loads((directory/'run.json').read_text())['status']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiments/waterbirds-shortcutminer-background-formal.json')
    parser.add_argument('--run-id')
    parser.add_argument('--live',action='store_true',help='Execute real five-Agent workflow and fresh quantitative training')
    parser.add_argument('--resume',action='store_true',help='Resume this run using its own persisted tools and checkpoints')
    args=parser.parse_args()
    if args.resume:
        if not args.live or not args.run_id: parser.error('--resume requires --live and an existing --run-id')
        resume(args.config,args.run_id)
    else:
        execute(args.config,args.run_id or 'waterbirds-shortcutminer-'+datetime.now().strftime('%Y%m%d-%H%M%S'),live=args.live)


def resume(config_path,run_id):
    directory=ROOT/'runs'/run_id
    receipt=json.loads((directory/'core_execution_receipt.json').read_text(encoding='utf-8'))
    if sha(Path(config_path))!=receipt['config_sha256']: raise RuntimeError('Resume configuration changed')
    if json.loads((directory/'run.json').read_text(encoding='utf-8'))['status']=='COMPLETED':
        raise RuntimeError('Run is already completed')
    config=json.loads(Path(config_path).read_text(encoding='utf-8')); root,data=prepare(config)
    if data!=receipt['data_view']: raise RuntimeError('Resume data identity changed')
    index=len(list(directory.glob('resume_execution_receipt_*.json')))+1
    stem=f'resume_{index:02d}'
    source=source_identity(require_clean=False)
    weights={str(p.relative_to(directory)).replace('\\','/'):sha(p) for p in directory.glob('models/task/**/*.npz')}
    write(directory/f'resume_execution_receipt_{index:02d}.json',{
        'source':source,'source_archive':stem+'_source_snapshot.zip',
        'source_archive_sha256':archive_source(directory/(stem+'_source_snapshot.zip'),source['source_tree_sha256']),
        'original_source_archive_sha256':receipt['source_archive_sha256'],'config_sha256':receipt['config_sha256'],
        'created_at':datetime.now(timezone.utc).isoformat(),'existing_weights_sha256':weights,
        'reason':'Resume failed workflow after protocol/artifact repair; keep original training provenance and existing models'})
    with formal_credentials():
        text=load_model_runtime_config(config['text_model_config']); vlm=load_model_runtime_config(config['vlm_model_config']).model_copy(update={'max_tokens':8192})
        tools=build_local_tool_runtimes(run_id=run_id,dataset_root=root,runs_root=ROOT/'runs',
            restore_persisted_task_results=True,max_tool_calls_per_run=150,max_tool_calls_per_task=40)
        for agent in tools:
            tools[agent].budget=ToolBudget(max_tool_calls=40,timeout_seconds=300,
                experiment_timeout_seconds={'task_model_training':None,'task_model_evaluation':1800,'ood_evaluation':1800},
                max_selected_samples=data['partition_counts']['ood'])
        text_client=OpenAICompatibleClient(text,runs_root=ROOT/'runs'); vlm_client=OpenAICompatibleClient(vlm,runs_root=ROOT/'runs')
        resume_run(run_id,runs_root=ROOT/'runs',agent_client=text_client,
            qualitative_client=QualitativeMultimodalClient(vlm_client,root),tool_runtimes=tools,
            semantic_inspector=OpenAICompatibleSemanticVisualInspector(vlm_client))
        for ref,expected in weights.items():
            if sha(directory/ref)!=expected: raise RuntimeError('Resume changed an existing model checkpoint')
        print(json.dumps({'run_id':run_id,'status':json.loads((directory/'run.json').read_text())['status']}),flush=True)

if __name__=='__main__': main()
