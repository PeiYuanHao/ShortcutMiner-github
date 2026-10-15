"""Opt-in training-only semantic coverage, with cached 4/8/12-per-class looks."""
import hashlib
import json
from pathlib import Path
from random import Random
from typing import Literal

from pydantic import Field
from shortcutminer.protocol.identify import StrictProtocolModel, NonEmptyText
from shortcutminer.protocol.visual_discovery import SemanticInspectionRequest, SemanticInspectionSample, SemanticObservation

POLICY = 'class-representative-v1'


class CoverageDecision(StrictProtocolModel):
    decision: Literal['stop', 'expand']
    rationale: NonEmptyText
    unresolved_observation_questions: list[NonEmptyText] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(min_length=1)


def nested_groups(observations, seed):
    groups = {}
    for item in observations:
        if item.decoded:
            groups.setdefault(item.group, []).append(item)
    for group, rows in groups.items():
        rows.sort(key=lambda r:r.sample_reference)
        local_seed = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16],16)
        Random(local_seed).shuffle(rows)
    return dict(sorted(groups.items()))


def inspect_representative(inspector, request, observations):
    """No evaluation paths or model outcomes; decisions concern observation coverage only."""
    client = getattr(inspector, 'coverage_client', None)
    if client is None:
        raise ValueError('Representative inspection requires an explicit coverage decision client')
    groups = nested_groups(observations, 2026)
    if not groups or 'ungrouped' in groups:
        raise ValueError('Representative semantic coverage requires explicit class groups')
    runs_root = getattr(inspector, 'coverage_runs_root', None)
    directory = Path(runs_root)/request.run_id/'identify/representative-coverage' if runs_root else None
    if directory: directory.mkdir(parents=True,exist_ok=True)
    source_identity={'policy':POLICY,'sources':[
        {'sample_id':r.sample_id,'reference':r.sample_reference,'group':g,
         'sha256':hashlib.sha256((Path(request.dataset.root)/r.sample_reference).read_bytes()).hexdigest()}
        for g,rows in groups.items() for r in rows]}
    if directory:
        path=directory/'source-identity.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=source_identity:
                raise ValueError('Representative coverage source input changed')
        else:path.write_text(json.dumps(source_identity,indent=2),encoding='utf-8')
    cumulative=[]; seen=set(); history=[]
    for look in (4,8,12):
        new=[r for rows in groups.values() for r in rows[:look] if r.sample_id not in seen]
        if not new: break
        for start in range(0,len(new),24):
            batch=new[start:start+24]
            identity={'policy':POLICY,'samples':[
                {'id':r.sample_id,'ref':r.sample_reference,'sha256':hashlib.sha256((Path(request.dataset.root)/r.sample_reference).read_bytes()).hexdigest()}
                for r in batch]}
            digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            path=directory/f'batch-{digest}.json' if directory else None
            if path and path.exists():
                result=json.loads(path.read_text(encoding='utf-8'))
                if result['input']!=identity: raise ValueError('Representative inspection identity changed')
                parsed=[SemanticObservation.model_validate(v) for v in result['observations']]
            else:
                parsed=inspector.inspect(dataset_root=request.dataset.root,request=SemanticInspectionRequest(
                    run_id=request.run_id,dataset_name=request.dataset.name,
                    samples=[SemanticInspectionSample(sample_id=r.sample_id,sample_reference=r.sample_reference) for r in batch]))
                parsed=[SemanticObservation.model_validate(v) for v in parsed]
                if path: path.write_text(json.dumps({'input':identity,'observations':[r.model_dump(mode='json') for r in parsed]},indent=2),encoding='utf-8')
            allowed={(r.sample_id,r.sample_reference) for r in batch}
            if any((r.sample_id,r.sample_reference) not in allowed for r in parsed):
                raise ValueError('Semantic observation outside its approved training batch')
            cumulative.extend(parsed)
            seen.update(r.sample_id for r in batch)
        coverage={g:sum(r.sample_id in seen for r in rows) for g,rows in groups.items()}
        context={'run_id':request.run_id,'mode':'discovery_coverage_review','policy':POLICY,
            'task':request.task.model_dump(mode='json'),'look_per_class':look,
            'observed_count':len(seen),'coverage_per_class':coverage,
            'group_by_sample':{r.sample_id:g for g,rows in groups.items() for r in rows if r.sample_id in seen},
            'observations':[r.model_dump(mode='json') for r in cumulative], 'previous_decisions':history,
            'response_schema':CoverageDecision.model_json_schema(),
            'scope':'Training semantic coverage only. Stop when observations suffice to formulate scoped hypotheses with support counts and confounds; expand only for specific unresolved observations or unstable apparent associations. No significance test, shortcut verdict, model effect or dataset identity inference. Never seek a desired positive result.'}
        digest=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
        path=directory/f'decision-{look}.json' if directory else None
        if path and path.exists():
            saved=json.loads(path.read_text(encoding='utf-8'))
            if saved['input_hash']!=digest: raise ValueError('Coverage decision input changed')
            decision=CoverageDecision.model_validate(saved['decision'])
        else:
            method=getattr(client,'invoke_agent_validated',None)
            kwargs={'system_prompt':'You are Identify reviewing training-only observation coverage. Return CoverageDecision JSON using the provided schema. This is an observation-budget decision, never a shortcut certification.'}
            raw=method('identify',context,validator=lambda v:CoverageDecision.model_validate(v),**kwargs) if method else client.invoke_agent('identify',context,**kwargs)
            decision=CoverageDecision.model_validate(raw)
            if path: path.write_text(json.dumps({'input_hash':digest,'decision':decision.model_dump(mode='json')},indent=2),encoding='utf-8')
        history.append({'look_per_class':look,'coverage_per_class':coverage,'decision':decision.model_dump(mode='json')})
        print(json.dumps({'stage':'representative_discovery','look':look,'observed':len(seen),'decision':decision.decision}),flush=True)
        if decision.decision=='stop': break
    receipt={'policy':POLICY,'scope':'training_only','numeric_decoded_count':sum(r.decoded for r in observations),
        'semantic_sample_count':len(seen),'semantic_sample_ids':sorted(seen),'looks':history,
        'budget_exhausted':bool(history and history[-1]['look_per_class']==12 and history[-1]['decision']['decision']=='expand'),
        'limitations':['Coverage decisions do not establish label independence, semantic preservation or model reliance.']}
    if directory: (directory/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return cumulative
