"""Opt-in candidate-complete-v1. Legacy experiment entry points are unchanged."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import run_autonomous_image_audit as legacy

from shortcutminer.audit import build_audit_run_request
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.experiment.candidate_validation import (
    POLICY, freeze_candidate_measurements, prepare_candidate_complete_view,
    label_group_preparation, write_candidate_dispositions,
)
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.protocol.model import ModelRuntimeConfigSet
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runtime.evidence_context import EvidenceContextClient
from shortcutminer.experiment.intervention_alignment import VERSION as ALIGNMENT_POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--training-input', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--runs-root', type=Path, default=legacy.ROOT/'runs')
    parser.add_argument('--vlm-max-tokens', type=int, default=8192)
    parser.add_argument('--isolated-discovery', action='store_true')
    parser.add_argument('--validation', action='store_true', required=True)
    parser.add_argument('--representative-review', action='store_true', help='Opt in to class-representative-v1 observation and training-construction-review-v2; legacy defaults unchanged')
    args = parser.parse_args()
    if not 2048 <= args.vlm_max_tokens <= 32768:
        parser.error('VLM output budget must be2048..32768')
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    if profile['promotion_policy']['strategy'] != 'budgeted_paired_probe':
        parser.error('candidate-complete-v1 requires the existing budgeted paired probe policy')
    knowledge = json.loads((legacy.ROOT/'configs/knowledge/research-policy-v1.json').read_text(encoding='utf-8'))
    if args.isolated_discovery: knowledge['research']['include_learned'] = False
    public = legacy.public_rows(args.input)
    if any(r['split'] != 'train' for r in public): raise ValueError('Discovery must be train only')
    with legacy.formal_credentials():
        text = legacy.load_model_runtime_config(legacy.ROOT/profile['text_model_config'])
        vlm = legacy.load_model_runtime_config(legacy.ROOT/profile['vlm_model_config']).model_copy(update={'max_tokens':args.vlm_max_tokens})
        models = ModelRuntimeConfigSet(shared=text, per_agent={'qualitative':vlm})
        client = EvidenceContextClient(text, runs_root=args.runs_root)
        vision = legacy.OpenAICompatibleClient(vlm, runs_root=args.runs_root, transport=legacy.UrllibStreamingJsonTransport())
        request = build_audit_run_request(AuditRequest(dataset_path=str(args.input), metadata_file='metadata.csv',
            label_information={'source':'metadata_column','column':'y'}, dataset_description=args.task),
            runs_root=args.runs_root, run_id=args.run_id+'-discovery', model_runtime=models)
        value = request.model_dump(mode='json')
        value['identify_config'] = {'budget':{'max_candidates':5,'max_samples':40,'max_dataset_entries':20000,'max_metadata_rows':10000},
            'visual_discovery':{'enabled':True,'sample_id_column':'img_id','sample_path_column':'img_filename','split_column':'split','selected_split':'train','max_samples':40,'semantic_enabled':True},
            'research_policy':knowledge['research']}
        value['qualitative_config'] = label_group_preparation('y')
        if args.representative_review:
            if len(public)>10000:raise ValueError('Representative discovery exceeds the 10,000-image numeric budget')
            value['identify_config']['budget']['max_samples']=len(public)
            value['identify_config']['visual_discovery'].update(sampling_strategy='class_representative_v1',
                max_samples=len(public),max_total_inspection_bytes=500_000_000)
        request = RunRequest.model_validate(value)
        initial = legacy.execute(request, client, vision, args.input, args.runs_root, vlm)
        report = json.loads((initial/'identify/output.json').read_text(encoding='utf-8'))
        observations = [json.loads(f.read_text(encoding='utf-8'))['payload'] for f in sorted((initial/'qualitative').glob('*/result.json'))]
        control = args.runs_root/(args.run_id+'-construction')
        plan, plan_receipt = freeze_candidate_measurements(client, root=args.input, candidates=report['candidates'],
            observations=observations, receipt_path=control/'candidate-measurements.json', prompt=load_agent_workspace_prompt('quantitative'))
        resources = SimpleNamespace(model_metadata_file='metadata.csv', split_column='split', train_split_values=['train'],
            sample_path_column='img_filename', max_train_samples=len(public))
        accepted = {}; outcomes = {}
        # Planning cap matches existing candidate probe cap, never model outcome feedback.
        cap = min(3, profile['promotion_policy']['max_probe_candidates'])
        planned = 0
        specs = {s.candidate_id:s for s in plan.measurements}
        for candidate in report['candidates']:
            cid = candidate['candidate_id']; resolution = plan_receipt['resolutions'][cid]
            if resolution['status'] != 'ready_for_cheap_checks':
                outcomes[cid] = {'status':'no_executable_measurement','reason':resolution['status']}; continue
            if planned >= cap:
                outcomes[cid] = {'status':'planning_budget_exhausted','reason':'No construction attempted; cheap checks remain enabled.'}; continue
            planned += 1; spec = specs[cid]
            receipt = legacy.select_autonomous_construction(client, vision, run_id=args.run_id+'-construction-'+cid,
                candidates=[candidate], observations=[o for o in observations if o.get('candidate_id')==cid],
                descriptor_keys=[spec.measurement_key], resources=resources, root=args.input,
                receipt_path=control/cid/'construction.json', task_description=args.task,
                training_diagnostics={'measurement':spec.model_dump(mode='json'), 'resolution':resolution},
                measurement_binding=spec.model_dump(mode='json'),
                quality_policy='training-construction-review-v2' if args.representative_review else None)
            outcomes[cid] = {'status':receipt['status'],'receipt_hash':receipt['receipt_hash'],
                'reason':receipt.get('reason','Training-only paired review accepted.')}
            if receipt['status']=='selected': accepted[cid] = receipt
        # Even zero accepted OOD recipes must not suppress all feasible cheap checks.
        view = args.runs_root/(args.run_id+'-input')
        binding = prepare_candidate_complete_view(plan_receipt, accepted, discovery_root=args.input,
            training_root=args.training_input, view=view)
        options = {k:v for k,v in profile['execution_options'].items() if k not in {'ood_specs','ood_recipe_selection','background_ood_specs','color_ood_specs','patch_ood_specs','visual_ood_specs','visual_measurement_bindings','excluded_scientific_fields'}}
        counts = binding['split_counts']
        options.update(model_metadata_file='model_metadata.csv',sample_id_column='img_id',sample_path_column='img_filename',
            split_column='split', train_split_values=['train'],validation_split_values=['val'],test_split_values=['id'],
            evaluation_split_values=['id'],max_train_samples=counts['train'],max_samples=counts['id'],max_intervention_samples=counts['id'],
            association_variable_mode='numeric_categorical',visual_measurement_bindings=binding['visual_measurement_bindings'],
            visual_ood_specs={f:o for f,o in binding['structured_specs'].items() if o['type']!='global_color'},
            color_ood_specs={f:o for f,o in binding['structured_specs'].items() if o['type']=='global_color'})
        request = build_audit_run_request(AuditRequest(dataset_path=str(view),metadata_file='discovery_metadata.csv',
            label_information={'source':'metadata_column','column':'y'}, execution_options=options,
            intake_budget={'max_entries':20000},dataset_description=args.task+'; Candidate-complete training-only measurement planning and Agent-owned interventions; model reliance requires paired evidence.'),
            runs_root=args.runs_root, run_id=args.run_id, model_runtime=models)
        value = request.model_dump(mode='json')
        value['identify_config'] = deepcopy(json.loads((initial/'frozen_execution_config.json').read_text(encoding='utf-8'))['identify_config'])
        value['identify_config']['visual_discovery']['semantic_enabled'] = False
        value['identify_config']['research_policy'] = {**knowledge['research'],'enabled':False,'include_learned':False}
        value['qualitative_config'] = label_group_preparation('y')
        value['executive_config'] = {'audit_options':{'validation_depth':'deep','max_samples_per_task':counts['id'],
            'qualitative_metadata_policy':'candidate-bound-v2','required_quantitative_checks':['association_test']}}
        value['metadata'].update(knowledge_policy=knowledge['learning'],promotion_policy=profile['promotion_policy'],
            source_policy='user_authorized_current_source_snapshot',candidate_validation_policy=POLICY,
            measurement_alignment_policy=ALIGNMENT_POLICY,
            measurement_plan_hash=plan_receipt['record_hash'],construction_outcomes=outcomes,
            discovery_report_sha256=legacy.sha(initial/'identify/output.json'),data_view_input_hash=legacy.sha(view/'binding_manifest.json'),
            paper_protocol={'confirmatory_blind_evidence':False,'isolated_discovery':args.isolated_discovery,'hidden_truth_used':False,
                'intervention_owner':'quantitative_agent','actual_paired_visual_review':True})
        if args.representative_review:
            value['metadata'].update(discovery_coverage_policy='class-representative-v1',
                construction_quality_policy='training-construction-review-v2')
        request = RunRequest.model_validate(value)
        raw = json.loads((initial/'identify/raw_output.json').read_text(encoding='utf-8'))
        directory = legacy.execute(request,client,vision,view,args.runs_root,vlm,legacy.FrozenIdentifyReplay(raw))
        dispositions = write_candidate_dispositions(directory,plan_receipt,outcomes)
        print(json.dumps({'run_id':args.run_id,'status':'COMPLETED','directory':str(directory),'candidate_dispositions':dispositions}),flush=True)


if __name__=='__main__': main()
