"""Lossless E4 IO encoding for supplied image descriptors, not new evidence."""
from copy import deepcopy
import math
from shortcutminer.inspection.image import flatten_image_descriptors
from shortcutminer.protocol.qualitative import QualitativeAgentDraft


def encode_qualitative_descriptors(output, payload):
    """Keep declared metadata; encode image facts in the proper existing schema.

    Aliases require unique numeric identity across all currently selected samples.
    Other supplied image descriptors are retained as observations. Unknown fields,
    unavailable samples and bad references are left untouched for normal rejection.
    No finding, assessment, counterevidence, confidence or existing index changes.
    """
    value=deepcopy(output);changes=[]
    if 'tool_calls' in value or not isinstance(value.get('metadata_evidence'),list):return value,changes
    QualitativeAgentDraft.model_validate(value)
    inputs=payload.get('request',{}).get('task',{}).get('inputs',{})
    if not inputs:return value,changes
    allowed={inputs.get('label_column'),*inputs.get('metadata_fields',[])}-{None}
    samples=payload.get('evidence_context',{}).get('selected_samples',[])
    by_id={s['sample_id']:s for s in samples}
    facts={s['sample_id']:flatten_image_descriptors(s.get('visual_descriptors',{}),file_size_bytes=s.get('byte_size')) for s in samples}
    kept=[]
    for index,item in enumerate(value['metadata_evidence']):
        field=item.get('field');refs=item.get('sample_refs',[])
        if field in allowed or not refs or any(ref not in by_id or ref not in value.get('sample_ids',[]) for ref in refs):
            kept.append(item);continue
        if any(field not in facts[ref] or not by_id[ref].get('sample_available') or by_id[ref].get('inspection_status') not in ('completed','limited') for ref in refs):
            kept.append(item);continue
        # Numeric identity is an encoding alias only, never an inferred association.
        aliases=[]
        for target in sorted(allowed):
            if target==inputs.get('label_column') or len(samples)<2:continue
            pairs=[(s.get('metadata',{}).get(target),facts[s['sample_id']].get(field)) for s in samples]
            if all(isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and math.isfinite(a) and a==b for a,b in pairs) and len({b for _,b in pairs})>=2:
                aliases.append(target)
        if len(aliases)==1:
            kept.append({**item,'field':aliases[0]})
            changes.append({'source_index':index,'source_field':field,'action':'exact_numeric_alias','target_field':aliases[0]})
        elif isinstance(value.get('observations'),list):
            start=len(value['observations'])
            for ref in refs:
                value['observations'].append({'sample_ref':ref,
                    'observation':'Supplied image descriptor '+field+'; original referenced samples '+str(refs)+': '+item['observation'],
                    'inspection_quality':'sufficient' if by_id[ref]['inspection_status']=='completed' else 'limited'})
            changes.append({'source_index':index,'source_field':field,'action':'supplied_descriptor_to_observations','observation_indexes':list(range(start,len(value['observations'])))})
        else:kept.append(item)
    value['metadata_evidence']=kept
    return value,changes
