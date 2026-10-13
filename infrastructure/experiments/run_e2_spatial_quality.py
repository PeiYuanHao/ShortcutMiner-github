"""Opt-in training-only geometry acceptance; same held-out reuse is development only."""
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Literal
import numpy as np
from PIL import Image
from pydantic import Field, ValidationError

ROOT=Path(r'E:\ShortcutMiner-github'); sys.path.insert(0,str(ROOT))
from shortcutminer.protocol.identify import StrictProtocolModel, CandidateId, NonEmptyText
from shortcutminer.intervention.protocol import SpatialTransform
from shortcutminer.intervention.engine import InterventionEngine, _region_mask
from shortcutminer.inspection.image import inspect_image_file, flatten_image_descriptors, load_rgb_image_file
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.experiment.loader import load_model_runtime_config
from shortcutminer.experiment.waterbirds_paper import source_identity, archive_source, write, sha
from shortcutminer.audit import build_audit_run_request, persist_audit_request
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.protocol.model import ModelRuntimeConfigSet
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runner import run, resume_run, build_local_tool_runtimes
from shortcutminer.runtime.formal_environment import formal_credentials
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient, UrllibStreamingJsonTransport
from shortcutminer.runtime.qualitative_multimodal import QualitativeMultimodalClient
from shortcutminer.tooling.runtime import ToolBudget

INITIAL=ROOT/'runs/e2-a-v2-blind-discovery-20261007'
INPUT=Path(r'E:\datasets\E2-A\e2-case-a-input-v2')
TRAINING=Path(r'E:\datasets\E2-A\e2-case-a-training-v2')
VIEW=ROOT/'runs/e2-a-v31-quality-development-input'
CONTROL=ROOT/'runs/e2-a-v31-quality-control'; CONTROL.mkdir(parents=True,exist_ok=True)
RUN_ID='e2-a-v31-quality-development-validation'
DESCRIPTORS=['dimensions','intensity','rgb_stats','saturation','entropy','edge_density','spatial_stats']

class AutonomousProposal(StrictProtocolModel):
    status: Literal['selected','deferred']
    candidate_id: CandidateId
    descriptor_key: NonEmptyText
    transformation: SpatialTransform | None
    rationale: NonEmptyText
    intended_invariants: list[NonEmptyText] = Field(min_length=1)
    semantic_risks: list[NonEmptyText] = Field(min_length=1)
    processing_controls: list[NonEmptyText] = Field(min_length=1)
    limitations: list[NonEmptyText] = Field(min_length=1)

def rows(root):
    with (root/'metadata.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream); assert reader.fieldnames==['img_id','img_filename','y','split']
        return list(reader)

def descriptors(path):
    r=inspect_image_file(path,requested_descriptors=DESCRIPTORS,max_file_bytes=1_000_000,max_pixels=1024,max_frames=1)
    assert r.decoded
    return flatten_image_descriptors(r.descriptors,file_size_bytes=r.byte_size)

def propose(client):
    assert json.loads((INITIAL/'run.json').read_text())['status']=='COMPLETED'
    report=json.loads((INITIAL/'identify/output.json').read_text())
    frozen_hash=sha(INITIAL/'identify/output.json')
    freeze=CONTROL/'v31_candidate_freeze.json'
    if freeze.exists(): assert json.loads(freeze.read_text())['candidate_report_sha256']==frozen_hash
    else: write(freeze,{'candidate_report_sha256':frozen_hash,'scope':'training_only_before_proposal_or_heldout_access'})
    observations=[]
    for candidate in report['candidates']:
        p=INITIAL/'qualitative'/f"TASK-{candidate['candidate_id']}-QL"/'result.json'
        if p.exists(): observations.append(json.loads(p.read_text())['payload'])
    profile=json.loads((INITIAL/'identify/visual_discovery.json').read_text())
    keys=sorted(profile['sample_observations'][0]['descriptors']['values'])
    sums={str(y):np.zeros((16,16)) for y in (0,1)}; counts=Counter()
    for row in rows(INPUT):
        assert row['split']=='train', 'Proposal diagnostics accept training rows only'
        rgb=load_rgb_image_file(INPUT/row['img_filename']).rgb.astype(float)/255
        gray=rgb.mean(axis=2); edge=np.zeros((32,32))
        edge[:,1:]+=abs(np.diff(gray,axis=1)); edge[1:,:]+=abs(np.diff(gray,axis=0))
        sums[row['y']]+=edge.reshape(16,2,16,2).mean(axis=(1,3)); counts[row['y']]+=1
    grid=np.abs(sums['0']/counts['0']-sums['1']/counts['1'])
    from types import SimpleNamespace
    from shortcutminer.experiment.spatial_construction_quality import select_spatial_construction
    resources=SimpleNamespace(model_metadata_file='metadata.csv', split_column='split',
                              train_split_values=['train'], sample_path_column='img_filename')
    selected=select_spatial_construction(client, candidates=report['candidates'], observations=observations,
        descriptor_keys=keys, resources=resources, root=INPUT,
        receipt_path=CONTROL/'v31_spatial_quality_receipt.json',
        prompt=load_agent_workspace_prompt('quantitative'),
        training_diagnostics={'absolute_class_edge_difference_16x16':np.round(grid,3).tolist(),
                              'grid_definition':'Each cell is a 2x2 block of a 32x32 training image.'})
    proposal=AutonomousProposal.model_validate(selected.model_dump(mode='json',exclude={'target_box'}))
    saved=CONTROL/'v31_autonomous_proposal.json'
    receipt={'proposal':proposal.model_dump(mode='json'), 'target_box':selected.target_box.model_dump(),
             'frozen_discovery_sha256':frozen_hash, 'construction_quality_receipt_sha256':sha(CONTROL/'v31_spatial_quality_receipt.json'),
             'selection_owner':'quantitative_agent','heldout_seen':False,'truth_seen':False,'external_ood_used':False}
    if saved.exists(): assert json.loads(saved.read_text())==receipt
    else: write(saved,receipt)
    return proposal

def prepare(proposal):
    path=VIEW/'binding_manifest.json'
    if path.exists():
        saved=json.loads(path.read_text()); assert saved['proposal']==proposal.model_dump(mode='json')
        for ref,digest in saved['file_hashes'].items(): assert sha(VIEW/ref)==digest
        for ref,digest in saved['training_source_hashes'].items(): assert sha(TRAINING/ref)==digest
        return saved
    assert not VIEW.exists()
    public=rows(TRAINING); assert Counter(r['split'] for r in public)=={'train':3000,'val':600,'id':800}
    assert {p.relative_to(TRAINING).as_posix() for p in TRAINING.rglob('*') if p.is_file()}=={r['img_filename'] for r in public}|{'metadata.csv','TASK.md'}
    assert not any(p.name=='ood' for p in TRAINING.rglob('*'))
    VIEW.mkdir()
    original_train={r['img_id']:r for r in rows(INPUT)}
    feature='visual_measurement_'+proposal.candidate_id.lower()
    groups={s:set() for s in ('train','val','id')}; hashes={'metadata.csv':sha(TRAINING/'metadata.csv')}
    for row in public:
        source=(TRAINING/row['img_filename']).resolve(strict=True); source.relative_to(TRAINING.resolve())
        if row['split']=='train':
            assert row==original_train[row['img_id']] and sha(source)==sha(INPUT/row['img_filename'])
        target=VIEW/row['img_filename']; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
        rgb=load_rgb_image_file(source).rgb
        mask=_region_mask(proposal.transformation,rgb.shape[1],rgb.shape[0])
        assert 0<float(mask.mean())<=.5
        groups[row['split']].add(hashlib.sha256(rgb[~mask].tobytes()).hexdigest())
        row[feature]=descriptors(source)[proposal.descriptor_key]; hashes[row['img_filename']]=sha(source)
    for a,b in (('train','val'),('train','id'),('val','id')): assert groups[a].isdisjoint(groups[b])
    assert len(groups['id'])==800
    for name,subset in (('model_metadata.csv',public),('discovery_metadata.csv',[r for r in public if r['split']=='train'])):
        with (VIEW/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['img_id','img_filename','y','split',feature]); writer.writeheader(); writer.writerows(subset)
    manifest={'proposal':proposal.model_dump(mode='json'),'feature_field':feature,'truth_read':False,'external_ood_used':False,
        'proposal_frozen_before_heldout_access':True,'training_source_hashes':hashes,
        'file_hashes':{p.relative_to(VIEW).as_posix():sha(p) for p in VIEW.rglob('*') if p.is_file()}}
    write(path,manifest); return manifest

class FrozenIdentifyReplay:
    def invoke_agent(self,name,payload,*,system_prompt=None):
        assert name=='identify'
        return deepcopy(json.loads((INITIAL/'identify/raw_output.json').read_text()))

def main():
    with formal_credentials():
        text=load_model_runtime_config(ROOT/'configs/models/pilot_text.example.yaml')
        vlm=load_model_runtime_config(ROOT/'configs/models/pilot_vlm.example.yaml').model_copy(update={'max_tokens':8192})
        client=OpenAICompatibleClient(text,runs_root=ROOT/'runs')
        proposal=propose(client)
        if '--proposal-only' in sys.argv: return
        if '--development-validation' not in sys.argv: raise RuntimeError('Use --proposal-only or --development-validation explicitly')
        data=prepare(proposal); feature=data['feature_field']
        template=json.loads((ROOT/'configs/experiments/synthetic-corner-patch-shortcutminer-memory-v4.json').read_text())
        options={k:v for k,v in template['execution_options'].items() if k not in ('ood_specs','ood_recipe_selection','background_ood_specs','color_ood_specs','patch_ood_specs')}
        options.update(max_train_samples=3000,max_samples=800,max_intervention_samples=800,model_metadata_file='model_metadata.csv',
            test_split_values=['id'],evaluation_split_values=['id'],association_variable_mode='numeric_categorical',
            visual_measurement_bindings={proposal.candidate_id:feature},visual_ood_specs={feature:proposal.transformation.model_dump(mode='json')})
        assert not any(spec.get('type')=='provided_pair' for spec in options['visual_ood_specs'].values())
        request=build_audit_run_request(AuditRequest(dataset_path=str(VIEW),metadata_file='discovery_metadata.csv',
            label_information={'source':'metadata_column','column':'y'},execution_options=options,intake_budget={'max_entries':12000},
            dataset_description='Continuation of frozen training-only blind discovery. Quantitative selected and froze a spatial intervention using training evidence; no external OOD or hidden truth is available. Generated paired counterfactual images are model-dependence tests; semantic preservation of filled regions is unverified.'),
            runs_root=ROOT/'runs',run_id=RUN_ID,model_runtime=ModelRuntimeConfigSet(shared=text,per_agent={'qualitative':vlm}))
        payload=request.model_dump(mode='json')
        payload.update(identify_config={'budget':{'max_candidates':5,'max_samples':40,'max_dataset_entries':12000,'max_metadata_rows':10000},
            'visual_discovery':{'enabled':True,'sample_id_column':'img_id','sample_path_column':'img_filename','split_column':'split','selected_split':'train','max_samples':40,'semantic_enabled':False}},
            executive_config={'audit_options':{'validation_depth':'deep','max_samples_per_task':800,'qualitative_metadata_policy':'candidate-bound-v2',
                'required_quantitative_checks':['association_test','task_model_training','task_model_evaluation','ood_evaluation']}},
            qualitative_config={'preparation':{'max_evidence_records':40,'max_contact_sheet_samples':40,'sampling_strategy':'group_balanced','group_columns':['y']}})
        payload['metadata'].update(promotion_policy=template['promotion_policy'],source_policy='user_authorized_current_source_snapshot',
            paper_protocol={'experiment':'E2-A-v3-development', 'confirmatory_blind_evidence':False, 'construction_protocol':'spatial-quality-v3.1','initial_blind_run_id':INITIAL.name,'initial_discovery_replayed':True,
                'initial_blind_report_sha256':sha(INITIAL/'identify/output.json'),'intervention_owner':'quantitative_agent',
                'intervention_proposal_sha256':sha(CONTROL/'v31_autonomous_proposal.json'),'external_ood_provided':False,'truth_used':False},
            data_view_input_hash=sha(VIEW/'binding_manifest.json'))
        request=RunRequest.model_validate(payload); directory=ROOT/'runs'/RUN_ID; resume=(directory/'run.json').exists()
        if not resume:
            persist_audit_request(request); source=source_identity(require_clean=False)
            write(directory/'core_execution_receipt.json',{'source':source,'source_archive_sha256':archive_source(directory/'source_snapshot.zip',source['source_tree_sha256']),
                'control_script_sha256':sha(Path(__file__)),'binding_manifest_sha256':sha(VIEW/'binding_manifest.json'),
                'discovery_origin_sha256':sha(INITIAL/'identify/output.json'),'autonomous_proposal_sha256':sha(CONTROL/'v31_autonomous_proposal.json'),
                'scope':'actual_shortcutminer_agent_selected_generated_counterfactual','truth_read':False,'external_ood_used':False})
            write(directory/'frozen_execution_config.json',payload)
            write(directory/'autonomous_intervention_proposal.json',json.loads((CONTROL/'v31_autonomous_proposal.json').read_text()))
        else:
            assert json.loads((directory/'frozen_execution_config.json').read_text())==payload
            if json.loads((directory/'run.json').read_text())['status']=='COMPLETED': return
        tools=build_local_tool_runtimes(run_id=RUN_ID,dataset_root=VIEW,runs_root=ROOT/'runs',restore_persisted_task_results=resume,
            max_tool_calls_per_run=150,max_tool_calls_per_task=40)
        for runtime in tools.values():
            runtime.budget=ToolBudget(max_tool_calls=40,timeout_seconds=300,max_selected_samples=800,
                experiment_timeout_seconds={'task_model_training':None,'task_model_evaluation':1800,'ood_evaluation':1800})
        vlm_client=OpenAICompatibleClient(vlm,runs_root=ROOT/'runs',transport=UrllibStreamingJsonTransport())
        clients={'agent_client':client,'identify_client':FrozenIdentifyReplay(),
            'qualitative_client':QualitativeMultimodalClient(vlm_client,VIEW,max_images=40,contact_sheet=True,runs_root=ROOT/'runs'),'tool_runtimes':tools}
        print(json.dumps({'run_id':RUN_ID,'status':'VALIDATION_STARTING','resume':resume}),flush=True)
        if resume: resume_run(RUN_ID,runs_root=ROOT/'runs',**clients)
        else: run(request,**clients)
        print(json.dumps({'run_id':RUN_ID,'status':json.loads((directory/'run.json').read_text())['status']}),flush=True)

if __name__=='__main__': main()
