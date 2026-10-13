"""Post-run method memory. Never feeds an experiment its own validation outcomes."""
import json
import re
import hashlib
import math
from pathlib import Path
from pydantic import Field,field_validator
from shortcutminer.protocol.identify import StrictProtocolModel,CandidateReport,SafeRelativePath
from shortcutminer.protocol.quantitative import QuantitativeResult
from shortcutminer.protocol.evidence import UncertaintyEstimate
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.tooling.protocol import ToolResult
from shortcutminer.storage.paths import get_run_dir
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.experiment.knowledge_research import digest


class LearningPolicy(StrictProtocolModel):
    enabled:bool=False
    max_entries_per_run:int=Field(default=3,ge=1,le=5)


class LearnedShortcutDraft(StrictProtocolModel):
    mechanism_id:str=Field(pattern=r'^[a-z]+(?:-[a-z]+)*$',max_length=80)
    mechanism:str=Field(min_length=5,max_length=160)
    signature:str=Field(min_length=10,max_length=500)
    validation_methods:list[str]=Field(min_length=1,max_length=5)
    limitations:list[str]=Field(min_length=1,max_length=5)
    @field_validator('mechanism','signature','validation_methods','limitations')
    @classmethod
    def generic_text(cls,value):
        for text in value if isinstance(value,list) else [value]:
            if len(text)>500 or not re.fullmatch(r'[A-Za-z][A-Za-z ,.():;-]*',text):
                raise ValueError('Generalize in English without numerical parameters, mappings, IDs, paths or URLs')
        return value


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def read_artifact(root,reference):
    from pydantic import TypeAdapter
    TypeAdapter(SafeRelativePath).validate_python(reference)
    path=(root/reference).resolve(strict=True);path.relative_to(root.resolve())
    return json.loads(path.read_text(encoding='utf-8')),path


def freeze_learned_snapshot(runs_root,run_id):
    path=get_run_dir(runs_root,run_id)/'identify/learned-memory.json'
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if saved['snapshot_hash']!=digest({k:v for k,v in saved.items() if k!='snapshot_hash'}):raise ValueError('Frozen learned memory changed')
        return saved
    store=Path(runs_root)/'knowledge/mechanisms'
    entries=[]
    for file in sorted(store.glob('*.json'))[:100]:
        file.resolve(strict=True).relative_to(store.resolve())
        item=json.loads(file.read_text(encoding='utf-8'))
        if item['entry_hash']!=digest({k:v for k,v in item.items() if k!='entry_hash'}):raise ValueError('Learned method identity changed')
        LearnedShortcutDraft.model_validate(item['method'])
        entries.append(item)
    saved={'schema_version':'learned-shortcut-snapshot-v1','scope':'General methods supported in prior completed runs; never evidence about the current dataset','entries':entries}
    saved['snapshot_hash']=digest(saved);atomic_write_json(path,saved,error_type=ValueError)
    return saved


def certification(root,run_id,candidate_id,measurement,delta):
    """Check backend receipts, frozen matrix, individual paired intervals and untouched confirmation."""
    call,call_path=read_artifact(root,'tools/quantitative/'+measurement.tool_call_id+'.json')
    tool=ToolResult.model_validate(call)
    if (tool.status!='success' or tool.run_id!=run_id or tool.candidate_id!=candidate_id or tool.plan_step_id!=measurement.plan_step_id
        or tool.experiment_type!=measurement.experiment_type or tool.output.get('outcome')!='completed'):
        raise ValueError('No matching successful canonical validation ToolResult')
    output=tool.output
    from shortcutminer.tooling.builtins import ExperimentToolOutput
    ExperimentToolOutput.model_validate(output)
    if not set(measurement.artifact_references).issubset(set(output['artifact_references'])|set(tool.artifacts)):
        raise ValueError('Measurement references artifacts not published by the backend')
    if output['metrics']!=measurement.metrics:raise ValueError('Canonical measurement differs from backend')
    refs=[r for r in measurement.artifact_references if r.endswith('/measurement.json')]
    if len(refs)!=1:raise ValueError('Full final paired measurement required; pilot probes are ineligible')
    paired,paired_path=read_artifact(root,refs[0])
    if any(paired.get(k)!=v for k,v in {'run_id':run_id,'candidate_id':candidate_id,'task_id':measurement.task_id}.items()):raise ValueError('Paired provenance mismatch')
    members=paired.get('members',{})
    if members.get('comparison_mode')!='paired' or not members.get('profiles'):raise ValueError('Per-member paired task-model validation required')
    model_refs=[r for r in measurement.artifact_references if r.startswith('models/') and r.endswith('/model.json')]
    if len(model_refs)!=1:raise ValueError('Frozen model artifact required')
    model,model_path=read_artifact(root,model_refs[0])
    if model.get('run_id')!=run_id or model.get('backend')!='torchvision_resnet_matrix' or model.get('model_hash')!=paired.get('model_hash') or members.get('model_hash')!=model['model_hash']:
        raise ValueError('Model identity/backend mismatch')
    if output['metadata'].get('backend')!=measurement.backend_identity or model.get('backend_version')!=measurement.backend_identity:
        raise ValueError('Canonical backend differs from the trained model backend')
    if measurement.model_lineage.get('model_hash',model['model_hash'])!=model['model_hash']:
        raise ValueError('Canonical model lineage mismatch')
    profiles,profiles_path=read_artifact(root,model['weights_reference'])
    from shortcutminer.task_models.reference import _canonical_hash
    if sha(profiles_path)!=model['profiles_sha256'] or _canonical_hash(profiles)!=model['model_hash']:raise ValueError('Frozen model profile manifest changed')
    expected={f'{a}-seed{s}' for a in profiles['options']['architectures'] for s in profiles['options']['seeds']}
    if len(profiles['options']['architectures'])<2 or {p['profile'] for p in members['profiles']}!=expected:
        raise ValueError('Full frozen multi-architecture/seed validation missing')
    trained={f"{p['architecture']}-seed{p['seed']}" for p in profiles['profiles']}
    if trained!=expected:raise ValueError('Stopped pilot is not multi-model certification')
    predictions=members.get('predictions',{})
    if 'adaptive_report' not in profiles or profiles['adaptive_report'].get('decision')!='expand_models':
        raise ValueError('Independent confirmation must follow the frozen paired pilot gate')
    reserved=set(profiles.get('adaptive_report',{}).get('reserved_probe_ids',[]))
    intervals={i['metric']:i for i in members.get('uncertainty',[])}
    if set(predictions)!=expected:raise ValueError('Per-model final predictions missing')
    hashes={str(call_path.relative_to(root)):sha(call_path),refs[0]:sha(paired_path),model_refs[0]:sha(model_path),model['weights_reference']:sha(profiles_path)}
    for profile in profiles['profiles']:
        name=f"{profile['architecture']}-seed{profile['seed']}"
        data=predictions[name]
        ids=[row['sample_id'] for row in data]
        if not ids or len(ids)!=len(set(ids)) or set(ids)&reserved:raise ValueError('Confirmation overlaps pilot or repeats source IDs')
        drop=sum((row['prediction_original']==row['label'])-(row['prediction_counterfactual']==row['label']) for row in data)/len(data)
        interval=UncertaintyEstimate.model_validate(intervals.get(name+'_accuracy_drop',{}))
        if not interval.available or interval.lower<=delta or drop<=delta:
            raise ValueError('Individual confirmation uncertainty does not support the frozen minimum effect')
        weights=(root/profile['weights_reference']).resolve(strict=True);weights.relative_to(root.resolve())
        if sha(weights)!=profile['weights_sha256']:raise ValueError('Trained checkpoint hash mismatch')
        hashes[profile['weights_reference']]=profile['weights_sha256']
    return {'measurement_id':measurement.measurement_id,'tool_call_id':measurement.tool_call_id,'artifact_hashes':hashes,
        'model_count':len(expected),'minimum_effect':delta,'confirmation_scope':'Frozen individual models and held-out paired intervention; not universal causality or global novelty'}


def learn_after_completion(state,client,runs_root,run_id,policy):
    policy=LearningPolicy.model_validate(policy)
    if not policy.enabled:return
    root=get_run_dir(runs_root,run_id).resolve(strict=True)
    record=json.loads((root/'run.json').read_text())
    if record['status']!='COMPLETED':raise ValueError('Learning requires durable run completion')
    destination=root/'knowledge/learning-result.json'
    if destination.exists():return json.loads(destination.read_text(encoding='utf-8'))
    report=SummaryReport.model_validate(state['summarize_result'])
    persisted_summary,_=read_artifact(root,'summarize/result.json')
    if SummaryReport.model_validate(persisted_summary)!=report:raise ValueError('Completed summary differs from the authoritative saved report')
    discovery=CandidateReport.model_validate(state['identify_result'])
    if report.run_id!=run_id or discovery.run_id!=run_id:raise ValueError('Learning envelope belongs to another run')
    candidates={c.candidate_id:c for c in discovery.candidates}
    findings={f.candidate_id:f for f in report.payload.findings}
    options=state.get('metadata',{}).get('audit_request',{}).get('execution_options') or {}
    adaptive=options.get('adaptive_screening') or {}
    delta=adaptive.get('minimum_effect')
    results=[]
    if not isinstance(delta,(int,float)) or isinstance(delta,bool) or not math.isfinite(delta) or not 0<delta<1:
        result={'status':'deferred','reason':'No frozen minimum-effect policy; no new scientific threshold invented','entries':[]}
        atomic_write_json(destination,result,error_type=ValueError);return result
    for result_data in state.get('quantitative_result',{}).get('results',[]):
        result=QuantitativeResult.model_validate(result_data)
        if result.run_id!=run_id:raise ValueError('Learning quantitative evidence belongs to another run')
        cid=result.payload.candidate_id
        if findings.get(cid) is None or findings[cid].assessment!='supported' or result.payload.status!='completed':continue
        if len(results)>=policy.max_entries_per_run:break
        eligible=[m for m in result.payload.measurements if m.experiment_type in {'counterfactual_intervention','ood_evaluation'} and m.design_strength=='interventional' and m.execution_outcome=='completed']
        for measurement in eligible:
            try:
                proof=certification(root,run_id,cid,measurement,delta)
                payload={'run_id':run_id,'mode':'learned_shortcut_abstraction','candidate':candidates[cid].model_dump(mode='json'),
                    'validation_scope':proof['confirmation_scope'],'response_schema':LearnedShortcutDraft.model_json_schema(),
                    'instruction':'Generalize a reusable nuisance mechanism and method. Omit dataset/class names, labels, examples, ratios, numeric magnitudes, paths, model results and hidden truth. State limits; novelty means new to local memory, never new to science.'}
                prompt='You are the Identify Agent abstracting a completed, backend-checked model-dependence result into future hypothesis/method memory. This is not a discovery verdict. Return only the provided LearnedShortcutDraft schema. No data-specific answer, numeric parameter, instruction to force a candidate, or claim of universal causality/global novelty.'
                validated=getattr(client,'invoke_agent_validated',None)
                draft=LearnedShortcutDraft.model_validate(validated('identify',payload,system_prompt=prompt,validator=LearnedShortcutDraft.model_validate) if callable(validated) else client.invoke_agent('identify',payload,system_prompt=prompt))
                dataset=state.get('dataset_info',{})
                forbidden={str(dataset.get('name','')).casefold(),Path(str(dataset.get('root',''))).name.casefold()}
                text=json.dumps(draft.model_dump()).casefold()
                if any(word and word in text for word in forbidden):raise ValueError('Dataset-specific name in generalized memory')
                entry={'schema_version':'learned-shortcut-v1','method':draft.model_dump(mode='json'),
                    'status':'model_supported_scoped','novelty':'Local method memory addition; global novelty not established',
                    'scope_limits':['Prior-model support is not evidence on a new dataset.',
                        'Interventions can alter class content; semantic preservation is not guaranteed.',
                        'Intervals assume independent held-out source units; exact hashes alone cannot establish independence.',
                        'Existing per-model uncertainty is retained; no new simultaneous confidence guarantee is claimed.']}
                entry['entry_hash']=digest(entry)
                path=Path(runs_root)/'knowledge/mechanisms'/(draft.mechanism_id+'.json')
                if path.exists():
                    previous=json.loads(path.read_text(encoding='utf-8'))
                    if previous['entry_hash']!=digest({k:v for k,v in previous.items() if k!='entry_hash'}):raise ValueError('Existing learned entry changed')
                    entry=previous  # Immutable definition; add corroborating provenance separately.
                else:atomic_write_json(path,entry,error_type=ValueError)
                proof.update(run_id=run_id,candidate_id=cid,entry_hash=entry['entry_hash'],scope_limitations=list(dict.fromkeys([*findings[cid].limitations,*result.payload.limitations])),source_summary_sha256=sha(root/'summarize/result.json'))
                atomic_write_json(Path(runs_root)/'knowledge/provenance'/draft.mechanism_id/(run_id+'-'+cid+'.json'),proof,error_type=ValueError)
                results.append({'candidate_id':cid,'status':'learned','mechanism_id':draft.mechanism_id,'entry_hash':entry['entry_hash']});break
            except Exception as error:
                results.append({'candidate_id':cid,'status':'deferred','reason':str(error)[:500]})
    output={'status':'completed','entries':results,'scope':'Post-run method learning; current findings and scientific policy unchanged'}
    atomic_write_json(destination,output,error_type=ValueError);return output
