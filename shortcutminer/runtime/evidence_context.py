"""Opt-in model context delivery; canonical evidence and validators stay intact."""
from copy import deepcopy
import hashlib
import json

from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient

ARRAYS = {'pixel_evidence','inference_snapshots','reserved_probe_ids','probe_sample_ids',
    'sample_ids','selected_sample_ids','predictions_original','predictions_intervention',
    'predictions_control','logits_original','logits_intervention','logits_control'}
REFERENCES = {'artifacts','artifact_references','output_artifact_references'}


def validate_approved_probe_routing(payload, draft):
    """An accepted executable intervention must have a budget-gated probe plan."""
    from shortcutminer.tooling.agent_loop import ProviderContractValidationError
    decisions = {d['candidate_id']:d for d in draft.get('decisions',[])}
    required = {'task_model_training','task_model_evaluation','ood_evaluation'}
    for context in payload.get('executable_candidate_contexts',[]):
        if context.get('structured_intervention_spec_available') and required.issubset(context['allowed_quantitative_checks']):
            decision = decisions.get(context['candidate_id'],{})
            checks = (decision.get('quantitative') or {}).get('requested_checks',[])
            if not required.issubset(checks):
                raise ProviderContractValidationError('executive_context_invalid',
                    f"decisions.{context['candidate_id']}.quantitative.requested_checks")


def evidence_context(payload):
    original = json.dumps(payload, sort_keys=True, allow_nan=False)
    if len(original) < 200000:
        return payload, None
    omitted = []
    def walk(value, path='root'):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                limit = 16 if key in REFERENCES else 0
                if isinstance(item, list) and ((key in ARRAYS and len(item)>40) or (key in REFERENCES and len(item)>16)):
                    omitted.append({'path':path+'.'+key,'count':len(item),'retained':limit,
                        'sha256':hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()})
                    out[key] = [walk(x,path+'.'+key) for x in item[:limit]]
                else: out[key] = walk(item,path+'.'+key)
            return out
        if isinstance(value,list): return [walk(item,path+f'[{i}]') for i,item in enumerate(value)]
        return value
    compact = walk(deepcopy(payload))
    compact['evidence_context_delivery'] = {'policy':'evidence-context-v1','omitted_arrays':omitted,
        'instruction':'This is a model-context summary only. Omitted arrays are preserved in canonical tool JSON and are not empty/missing evidence. All metrics, uncertainty, checkpoint decisions and options remain. Do not invent predictions or final confirmation. References are representative; full lists remain on disk.'}
    encoded = json.dumps(compact,sort_keys=True,allow_nan=False)
    record = {'policy':'evidence-context-v1','original_characters':len(original),'compact_characters':len(encoded),
        'original_payload_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'delivered_payload_sha256':hashlib.sha256(encoded.encode()).hexdigest(),'omitted_arrays':omitted}
    return compact, record


class EvidenceContextClient(OpenAICompatibleClient):
    """Only the new entry opts in; request hashes cover the actual compact input."""
    def _delivery(self, agent, payload):
        if agent not in {'quantitative','summarize'}: return payload
        compact, record = evidence_context(payload)
        if record is not None:
            run_id = payload.get('run_id') or payload.get('request',{}).get('run_id')
            if run_id and self._runs_root is not None:
                from shortcutminer.storage.events import append_event
                append_event(self._runs_root, run_id=run_id, event='EVIDENCE_CONTEXT_DELIVERY',
                    stage=agent, details=record)
        return compact

    def invoke_agent(self, agent_name, payload, **kwargs):
        return super().invoke_agent(agent_name,self._delivery(agent_name,payload),**kwargs)

    def invoke_agent_validated(self, agent_name, payload, **kwargs):
        if agent_name == 'executive':
            original_validator = kwargs['validator']
            def validate(draft):
                original_validator(draft)
                validate_approved_probe_routing(payload,draft)
            kwargs['validator'] = validate
            kwargs['system_prompt'] = (kwargs.get('system_prompt') or '') + (
                ' In candidate-complete-v1, candidates with structured_intervention_spec_available=true '
                'and all three model checks allowed MUST include task_model_training, task_model_evaluation '
                'and ood_evaluation in their quantitative plan, along with caller-required cheap checks. '
                'These steps are still subject to the existing promotion/probe budget and sequential early stopping. '
                'Do not require model checks for candidates without an approved executable intervention.')
        return super().invoke_agent_validated(agent_name,self._delivery(agent_name,payload),**kwargs)
