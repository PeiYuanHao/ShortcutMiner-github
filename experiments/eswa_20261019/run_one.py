"""Bounded isolated ESWA arm, reusing the existing E4 scientific harness.

The only additional runtime changes are shared training-only cache identities,
disabled cross-run learning/retrieval, and an aggregate text+vision budget.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.e4_ablation.scripts import run_arm as arm
from experiments.e4_ablation.variants.policies import write
from shortcutminer.runtime.openai_compatible import PilotInvocationBudgetState
from shortcutminer.task_models.torchvision_matrix import TorchvisionResNetMatrix, BACKEND_VERSION
from shortcutminer.experiment.waterbirds_paper import sha, source_identity, archive_source


def cache_identity(paths,labels,val_paths,val_labels,options,initialization_hashes):
    value={'backend':BACKEND_VERSION,'training':[[sha(p),str(y)] for p,y in zip(paths,labels)],
        'validation':[[sha(p),str(y)] for p,y in zip(val_paths,val_labels)],
        'options':options,'initialization':initialization_hashes,
        'selection':'validation_overall_accuracy_then_earliest_epoch'}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest(),value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--case-id',required=True)
    parser.add_argument('--variant',choices=('full','single_agent','fixed_pipeline'),required=True)
    parser.add_argument('--phase',choices=('development','formal'),required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--shared-training-cache-root',type=Path,
        help='Explicit study cache root; content/label/recipe/initialization identities remain checked.')
    parser.add_argument('--token-budget',type=int,default=300000)
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--compact-role-prompts',action='store_true')
    parser.add_argument('--reserve-missing-text-usage',action='store_true')
    parser.add_argument('--qualitative-sample-cap',type=int,choices=(12,40),default=40)
    parser.add_argument('--qualitative-task-cap',type=int,choices=(1,5),default=5)
    parser.add_argument('--resource-policy-development-fork',action='store_true')
    parser.add_argument('--continue-development-from',type=Path,
        help='Reuse only completed training-only phases of a quota-failed development attempt; never formal outcomes.')
    args=parser.parse_args(); study=args.study_root.resolve()
    index=study/('DEVELOPMENT_INPUTS.json' if args.phase=='development' else 'FORMAL_PUBLIC_INPUTS.json')
    case=next(x for x in json.loads(index.read_text())['cases'] if x['case_id']==args.case_id)
    output=study/'audits'/args.phase/args.case_id/args.variant/f'attempt-{args.attempt:02d}'
    if output.exists(): raise ValueError('Attempt directory exists. Preserve it; use an explicit new attempt.')
    output.mkdir(parents=True)
    prior_tokens=0;previous=None;previous_receipt=None
    if args.continue_development_from is not None:
        if args.phase!='development':raise ValueError('Development continuation cannot supply formal results')
        previous=args.continue_development_from.resolve(strict=True)
        previous.relative_to(study/'audits/development')
        previous_receipt=json.loads((previous/'EXECUTION_RECEIPT.json').read_text())
        old_state=json.loads((previous/'arm-status.json').read_text())
        prior_run=previous/(args.case_id+'-'+args.variant.replace('_','-'))
        profile_matches=previous_receipt['profile_sha256']==sha(args.profile.resolve())
        if args.resource_policy_development_fork and not profile_matches:
            from experiments.eswa_20261019.resource_policy import single_probe_profile
            after=json.loads(args.profile.read_text(encoding='utf-8-sig'))
            for name in ('development-profile.json','one-construction-profile.json'):
                old_profile=study/'protocol'/name
                before=json.loads(old_profile.read_text(encoding='utf-8-sig'))
                legacy=json.loads(old_profile.read_text(encoding='utf-8-sig'))
                legacy['promotion_policy']['max_probe_candidates']=1
                if previous_receipt['profile_sha256']==sha(old_profile) and after in (single_probe_profile(before),legacy):
                    profile_matches=True
        if (previous_receipt['case_id']!=args.case_id or previous_receipt['variant']!=args.variant
            or not profile_matches
            or old_state['status']!='FAILED'
            or (old_state.get('stage') not in {'discovery','validation'})
            or (not args.resource_policy_development_fork and list(prior_run.glob('quantitative/*/result.json')))
            or (not args.resource_policy_development_fork and list(prior_run.glob('qualitative/*/result.json')))
            or (prior_run/'models').exists()):
            raise ValueError('Only the same profile and pre-validation development audit may continue')
        if sha(Path(case['training'])/'metadata.csv')!=case['metadata_sha256']:
            raise ValueError('Public source metadata changed')
        records=[json.loads(p.read_text()) for p in previous.glob('*/model_calls/*.json')
                 if not p.name.startswith(('telemetry','runtime_snapshot'))]
        alpha_guard=(args.resource_policy_development_fork and old_state.get('error_type')=='ValueError'
            and old_state.get('actual_tool_calls')==0 and
            'Adaptive alpha allocation and promotion candidate budgets differ' in
            (study/'logs/dev-03-resource-protocol.stderr.log').read_text(encoding='utf-8'))
        if not alpha_guard and not any(r.get('failure_category')=='budget_exceeded' or
                   (r.get('agent_name')=='executive' and r.get('failure_category')=='protocol_schema_failed') for r in records):
            raise ValueError('Continuation requires a recorded pre-model quota or routing-contract failure')
        prior_tokens=previous_receipt['known_tokens']
        if args.token_budget<=prior_tokens:raise ValueError('Continuation has no remaining quota')
        old_control=previous/(args.case_id+'-'+args.variant.replace('_','-')+'-construction')
        if old_control.exists():shutil.copytree(old_control,output/old_control.name,
            ignore=shutil.ignore_patterns('model_calls','events.jsonl'))
        old_memory=previous/'single-controller-memory.json'
        if args.variant=='single_agent' and old_memory.exists():
            if args.resource_policy_development_fork:
                from experiments.eswa_20261019.resource_policy import prevalidation_memory_for_constructions
                turns=json.loads(old_memory.read_text())
                measurements=json.loads((old_control/'candidate-measurements.json').read_text())
                from experiments.eswa_20261019.resource_policy import completed_discovery
                discovery=completed_discovery(previous,args.case_id+'-'+args.variant.replace('_','-')+'-discovery',study/'audits/development')/'identify/output.json'
                candidates=json.loads(discovery.read_text())['candidates']
                selected=[c['candidate_id'] for c in candidates if measurements['resolutions'][c['candidate_id']]['status']=='ready_for_cheap_checks'][:1]
                retained=prevalidation_memory_for_constructions(turns,set(selected))
                write(output/'single-controller-memory.json',retained)
                write(output/'MEMORY_FORK.json',{'original':str(old_memory),'original_sha256':sha(old_memory),
                    'retained_turns':len(retained),'original_turns':len(turns),'selected_construction_candidates':selected,
                    'scope':'New one-construction development protocol, fork before failed validation; original full memory retained at source.'})
            else:shutil.copyfile(old_memory,output/'single-controller-memory.json')
        write(output/'CONTINUATION.json',{'previous_attempt':str(previous),
            'previous_receipt_sha256':sha(previous/'EXECUTION_RECEIPT.json'),
            'scope':'Same development audit; completed discovery and accepted training-only constructions reused; incomplete construction is retried, no heldout scientific outcome exists.',
            'source_change':'Old phase artifacts retain their original source identity; new phases record the new source snapshot.',
            'prior_known_tokens':prior_tokens,'revised_cumulative_development_stop':args.token_budget})
    identity=source_identity(require_clean=False)
    write(output/'source-identity.json',{'source':identity,
        'archive_sha256':archive_source(output/'source.zip',identity['source_tree_sha256'])})
    Budget=PilotInvocationBudgetState
    if args.reserve_missing_text_usage:
        from experiments.eswa_20261019.usage_budget import ReservedUsageBudget,observed_transport
        Budget=ReservedUsageBudget
    shared=Budget(max_text_calls=200,max_vlm_calls=50,max_vlm_images=5000,
        max_text_per_run=200,max_vlm_per_run=50,max_vlm_images_per_run=5000,max_tokens=args.token_budget)
    if previous_receipt is not None:
        shared.known_tokens=prior_tokens
        shared.text_calls=previous_receipt['text_calls'];shared.vlm_calls=previous_receipt['vision_calls']
        shared.vlm_images=previous_receipt['submitted_images']
    if args.compact_role_prompts:
        from experiments.eswa_20261019.compact_roles import install,supply_schema,VERSION
        install()
    class SchemaDelivery:
        def _reserve_call(self,*a,**kw):
            if args.reserve_missing_text_usage:shared.begin_call()
            return super()._reserve_call(*a,**kw)
        def _invoke_structured(self,**kw):
            if args.compact_role_prompts:
                from copy import deepcopy
                original_prompt=kw.get('system_prompt')
                prompt=supply_schema(kw['agent_name'],kw['payload_for_hash'],original_prompt)
                if prompt!=original_prompt:
                    kw['messages']=deepcopy(kw['messages'])
                    for message in kw['messages']:
                        if message['role']=='system' and message['content']==original_prompt:
                            message['content']=prompt
                    kw['system_prompt']=prompt
            validator=kw.get('validator')
            if validator is not None:
                def observed_validator(raw):
                    try:return validator(raw)
                    except Exception as exc:
                        encoded=json.dumps(raw,sort_keys=True,default=str)
                        key=hashlib.sha256(encoded.encode()).hexdigest()[:20]
                        issues=exc.errors(include_input=False,include_context=False) if hasattr(exc,'errors') else []
                        write(output/'protocol_rejections'/(kw['agent_name']+'-'+key+'.json'),
                            {'agent':kw['agent_name'],'run_id':kw['run_id'],'error_type':type(exc).__name__,
                             'issues':issues,'provider_draft':raw,'validator_bypassed':False})
                        raise
                kw['validator']=observed_validator
            return super()._invoke_structured(**kw)
    class Text(SchemaDelivery,arm.TextClient):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self._pilot_budget_state=shared
            if args.reserve_missing_text_usage:self._transport=observed_transport(self._transport,shared)
        def invoke_agent_validated(self,agent_name,payload,**kw):
            if agent_name=='executive' and 'executable_candidate_contexts' in payload and args.qualitative_task_cap<5:
                from experiments.eswa_20261019.resource_policy import bound_visual_tasks
                payload=bound_visual_tasks(payload,args.qualitative_task_cap)
                original=kw['validator']
                def validate(draft):
                    original(draft)
                    if sum(d.get('qualitative') is not None for d in draft.get('decisions',[]))>args.qualitative_task_cap:
                        raise ValueError('Shared qualitative task allocation exceeded')
                kw['validator']=validate
            return super().invoke_agent_validated(agent_name,payload,**kw)
    class Vision(SchemaDelivery,arm.VisionClient):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self._pilot_budget_state=shared
            if args.reserve_missing_text_usage:self._transport=observed_transport(self._transport,shared)
    arm.TextClient=Text;arm.VisionClient=Vision
    load_original=arm.legacy.load_model_runtime_config
    def load_config(path):
        config=load_original(path)
        return config.model_copy(update={'timeout':120.0})
    arm.legacy.load_model_runtime_config=load_config
    original_restrict=arm.restrict_request
    def restrict(request,variant):
        request=original_restrict(request,variant)
        value=request.model_dump(mode='json')
        policy=value['identify_config'].setdefault('research_policy',{})
        policy.update(enabled=False,include_learned=False)
        value['metadata']['knowledge_policy']={'enabled':False,'max_entries_per_run':3}
        preparation=value['qualitative_config'].setdefault('preparation',{})
        preparation.update(max_evidence_records=args.qualitative_sample_cap,
            max_contact_sheet_samples=args.qualitative_sample_cap)
        value['metadata']['eswa_study']={'phase':args.phase,'private_truth_used':False,
            'role_prompt_protocol':VERSION if args.compact_role_prompts else 'original_workspace_documents',
            'qualitative_sample_cap':args.qualitative_sample_cap,
            'qualitative_task_cap':args.qualitative_task_cap,
            'missing_text_usage_policy':'conservative_reservation_v1' if args.reserve_missing_text_usage else 'fail_closed',
            'aggregate_known_token_stop':args.token_budget,'token_stop_can_overshoot_by_one_inflight_request':True,
            'shared_training_cache':'content,labels,validation,recipe,initialization; no candidate or heldout effects',
            'independent_audit_repetitions':1}
        return type(request).model_validate(value)
    arm.restrict_request=restrict
    if previous is not None:
        execute_original=arm.legacy.execute
        def execute_continuation(request,*a,**kw):
            if request.run_id.endswith('-discovery'):
                from experiments.eswa_20261019.resource_policy import completed_discovery
                return completed_discovery(previous,request.run_id,study/'audits/development')
            return execute_original(request,*a,**kw)
        arm.legacy.execute=execute_continuation
    original_fit=TorchvisionResNetMatrix.fit_matrix
    def fit(model,paths,labels,val_paths,val_labels,directory,fingerprint,*,probe_callback=None,shared_cache=None):
        from shortcutminer.experiment.waterbirds_paper import initialization
        init={arch:sha(initialization(arch)[0]) for arch in model.options.architectures}
        key,content=cache_identity(paths,labels,val_paths,val_labels,model.options.model_dump(mode='json'),init)
        cache=(args.shared_training_cache_root.resolve() if args.shared_training_cache_root else study/'shared_classifiers')/key
        record=cache/'TRAINING_IDENTITY.json'
        if record.exists() and json.loads(record.read_text())!=content:
            raise ValueError('Training cache identity mismatch')
        if not record.exists(): write(record,content)
        write(Path(directory)/'eswa-shared-training.json',{'identity':key,
            'cache':str(cache),'heldout_model_outcomes_used':False,'source_tree':identity['source_tree_sha256']})
        result=original_fit(model,paths,labels,val_paths,val_labels,directory,fingerprint,
            probe_callback=probe_callback,shared_cache=cache)
        from experiments.eswa_20261019.resource_policy import publish_probe_cache
        publish_probe_cache(model,cache,shared_cache)
        return result
    TorchvisionResNetMatrix.fit_matrix=fit
    original_argv=sys.argv
    sys.argv=[str(Path(arm.__file__)),'--variant',args.variant,
        '--input',case['discovery'],'--training-input',case['training'],
        '--profile',str(args.profile.resolve()),'--runs-root',str(output),
        '--run-id',args.case_id+'-'+args.variant.replace('_','-'),
        '--task',case['task'],'--protocol-revision','descriptor-codec-v3']
    started=time.time()
    try:
        arm.main()
    finally:
        sys.argv=original_argv
        write(output/'EXECUTION_RECEIPT.json',{'phase':args.phase,'case_id':args.case_id,
            'variant':args.variant,'started_unix':started,'finished_unix':time.time(),
            'wall_seconds':time.time()-started,'text_calls':shared.text_calls,
            'vision_calls':shared.vlm_calls,'submitted_images':shared.vlm_images,
            'known_tokens':shared.known_tokens,'usage_incomplete':shared.usage_incomplete,
            'prior_known_tokens':prior_tokens,'new_tokens':shared.known_tokens-prior_tokens,
            'aggregate_known_token_stop':args.token_budget,
            'source_tree_sha256':identity['source_tree_sha256'],
            'profile_sha256':sha(args.profile.resolve()),
            'role_prompt_protocol':VERSION if args.compact_role_prompts else 'original_workspace_documents',
            'qualitative_sample_cap':args.qualitative_sample_cap,
            'qualitative_task_cap':args.qualitative_task_cap,
            'unknown_token_reservations':getattr(shared,'unknown_token_reservations',[]),
            'reserved_unknown_tokens':getattr(shared,'reserved_unknown_tokens',0),
            'development_results_are_not_formal':args.phase=='development'})


if __name__=='__main__':main()
