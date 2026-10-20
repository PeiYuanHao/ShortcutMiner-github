"""Explicit input-only natural-language feedback ablation; canonical reviews stay intact."""
from copy import deepcopy
import json

VERSION='construction-and-qualitative-text-feedback-v1'
CONTEXT_KEYS=frozenset({'training_qualitative_observations','training_observations','visual_review',
    'quality_review_requires_revision','executive_review','qualitative_result'})
TEXT_KEYS=frozenset({'cue_change_observation','task_content_observation','artifacts_observation',
    'limitations','preservation_risks','rationale','summary','observation','interpretation',
    'description','explanation','overview','notes','evidence_summary'})
WITHHELD='[Natural-language semantic feedback withheld by frozen ablation; categorical and numerical evidence retained.]'


def mask_feedback(payload):
    changes=[]
    def walk(value,path='root',semantic=False):
        if isinstance(value,dict):
            semantic=semantic or value.get('source')=='qualitative'
            result={}
            for key,item in value.items():
                active=semantic or key in CONTEXT_KEYS
                if active and key in TEXT_KEYS and isinstance(item,(str,list)):
                    changes.append(path+'.'+key)
                    result[key]=WITHHELD if isinstance(item,str) else [WITHHELD for _ in item]
                else:result[key]=walk(item,path+'.'+key,active)
            return result
        if isinstance(value,list):return [walk(item,path+f'[{i}]',semantic) for i,item in enumerate(value)]
        return deepcopy(value)
    return walk(payload),changes


def mask_messages(messages):
    result=deepcopy(messages)
    for message in result:
        if message.get('role')!='user' or not isinstance(message.get('content'),str):continue
        try:value=json.loads(message['content'])
        except ValueError:continue
        if isinstance(value,dict):message['content']=json.dumps(mask_feedback(value)[0],sort_keys=True,separators=(',',':'))
    return result
