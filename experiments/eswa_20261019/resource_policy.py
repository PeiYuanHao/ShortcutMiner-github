"""Common pre-outcome per-audit qualitative task allocation; no truth access."""
from copy import deepcopy
import json
from pathlib import Path
import shutil


def publish_probe_cache(model,cache,destination):
    """Preserve backend artifact references when a study cache supplies training."""
    if destination is None:return
    from shortcutminer.experiment.waterbirds_paper import sha
    for profile in model.profiles:
        name=f"{profile['architecture']}-seed{profile['seed']}"
        source=Path(cache)/name;target=Path(destination)/name
        source.resolve(strict=True).relative_to(Path(cache).resolve())
        target.resolve().relative_to(Path(destination).resolve())
        record=json.loads((source/'profile.json').read_text())
        if sha(source/'best.npz')!=record['weights_sha256'] or record['weights_sha256']!=profile['weights_sha256']:
            raise ValueError('Probe cache weights changed')
        target.mkdir(parents=True,exist_ok=True)
        for filename in ('best.npz','profile.json'):shutil.copyfile(source/filename,target/filename)


def single_probe_profile(profile):
    value=deepcopy(profile)
    adaptive=value['execution_options']['adaptive_screening']
    adaptive['family_alpha']/=adaptive['candidate_budget']
    adaptive['candidate_budget']=1
    value['promotion_policy']['max_probe_candidates']=1
    return value


def completed_discovery(previous,run_id,development_root):
    """Resolve retained discovery through bounded, containment-checked continuations."""
    from shortcutminer.experiment.waterbirds_paper import sha
    seen=set();current=Path(previous).resolve()
    for _ in range(10):
        current.relative_to(Path(development_root).resolve())
        if current in seen:raise ValueError('Cyclic development continuation')
        seen.add(current);old=current/run_id
        if old.exists():
            if json.loads((old/'run.json').read_text())['status']!='COMPLETED':
                raise ValueError('Only a completed discovery can be reused')
            source=json.loads((old/'source_identity.json').read_text())
            if sha(old/'source_snapshot.zip')!=source['archive_sha256']:
                raise ValueError('Previous discovery source archive changed')
            return old
        current=Path(json.loads((current/'CONTINUATION.json').read_text())['previous_attempt']).resolve()
    raise ValueError('Development continuation depth exceeded')


def bound_visual_tasks(payload,cap):
    value=deepcopy(payload);contexts=value.get('executable_candidate_contexts',[])
    eligible=[c for c in contexts if c.get('allowed_qualitative_checks')]
    preferred=[c for c in eligible if c.get('structured_intervention_spec_available')]
    ordered=preferred+[c for c in eligible if c not in preferred]
    selected={c['candidate_id'] for c in ordered[:cap]}
    for context in contexts:
        if context['candidate_id'] not in selected:context['allowed_qualitative_checks']=[]
    value['shared_resource_policy']={'max_qualitative_tasks':cap,
        'selected_candidate_ids':sorted(selected),
        'rule':'Prefer training-approved structured constructions, then frozen candidate order; no heldout outcome or hidden truth.',
        'limitations':'Other candidates retain feasible quantitative checks and discovery observations, not completed qualitative validation.'}
    return value


def prevalidation_memory_for_constructions(turns,selected):
    """Fork a development controller before validation, retaining only budgeted construction turns."""
    output=[];routing_count=0;keep=True
    for turn in turns:
        response=turn.get('response',{})
        if turn.get('phase_adapter')=='executive' and 'decisions' in response:
            routing_count+=1
            if routing_count==2:break
        if turn.get('phase_adapter')=='quantitative' and 'candidate_id' in response and 'descriptor_key' in response:
            keep=response['candidate_id'] in selected
        if keep:output.append(deepcopy(turn))
    return output
