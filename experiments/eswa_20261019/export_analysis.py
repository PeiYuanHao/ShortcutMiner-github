"""Evidence-grounded export. No Agent calls, model fitting, fabricated adjudication or canonical rewrites."""
import argparse,csv,json,shutil,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.waterbirds_paper import sha,source_identity,archive_source

MAIN=ROOT/'results_eswa_single'


def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def csv_write(path,rows,fields):
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def paired_effect(record):
    metrics=record.get('metrics',{});parameters=record.get('parameters',{})
    mode=parameters.get('comparison_mode',record.get('details',{}).get('comparison_mode'))
    delta=metrics.get('accuracy_delta')
    if mode!='paired' or delta is None:return None
    interval=next((item for item in record.get('uncertainty',[]) if item.get('metric')=='accuracy_delta' and item.get('available')),None)
    return {'n_pairs':metrics.get('usable_pair_count',metrics.get('pair_count')),
        'id_accuracy':metrics.get('original_accuracy',metrics.get('id_accuracy')),
        'intervention_accuracy':metrics.get('counterfactual_accuracy',metrics.get('ood_accuracy')),
        'accuracy_drop':-delta,'lower_95':-interval['upper'] if interval else None,
        'upper_95':-interval['lower'] if interval else None,'interval_method':interval.get('method') if interval else None,
        'bootstrap_iterations':interval.get('bootstrap_iterations') if interval else None,
        'preservation_status':parameters.get('preservation_status'),'outcome':record.get('outcome'),
        'transformation_json':json.dumps(parameters.get('transformation'),sort_keys=True),
        'model_hash':metrics.get('model_hash')}


def usage_records(attempt,excluded_ids=frozenset()):
    records=[];seen=set()
    for path in sorted(attempt.glob('*/model_calls/*.json')):
        record=read(path,{})
        identifier=record.get('invocation_id')
        if not identifier or identifier in seen or identifier in excluded_ids:continue
        seen.add(identifier);records.append((path,record))
    actual=[(path,record) for path,record in records if record.get('status')!='budget_exceeded']
    known=sum(record.get('usage',{}).get('total_tokens') or 0 for _,record in actual if record.get('usage',{}).get('available'))
    unknown=[(path,record) for path,record in actual if not record.get('usage',{}).get('available')]
    return known,actual,unknown


def changed_text_leaves(before,after):
    if isinstance(before,str):return int(bool(before) and before!=after)
    if isinstance(before,list) and isinstance(after,list):return sum(changed_text_leaves(a,b) for a,b in zip(before,after))
    if isinstance(before,dict) and isinstance(after,dict):return sum(changed_text_leaves(value,after.get(key)) for key,value in before.items())
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='eswa-20261020-v1');args=parser.parse_args()
    output=MAIN/'delivery'/args.version;output.mkdir(parents=True,exist_ok=False)
    tables=output/'tables';tables.mkdir();(output/'figures').mkdir();(output/'reproducibility').mkdir()
    studies=[('main',MAIN,MAIN/'FORMAL_TASK_STATUS.csv'),
        ('controlled',MAIN/'extensions/fixed-candidate-v1',MAIN/'extensions/fixed-candidate-v1/TASK_STATUS.csv'),
        ('pet',MAIN/'extensions/pet-v1',MAIN/'extensions/pet-v1/TASK_STATUS.csv')]
    tasks=[];candidates=[];effects=[];tools=[];screens=[];masks=[];references=[];truth=[]
    for study,root,status in studies:
        with status.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        for row in rows:
            if row['status'] not in {'COMPLETED','FAILED_RETAINED'}:raise ValueError('Agent audit not terminal')
            attempt=root/'audits/formal'/row['case_id']/row['variant']/'attempt-01'
            run=attempt/(row['case_id']+'-'+row['variant'].replace('_','-'))
            receipt=read(attempt/'EXECUTION_RECEIPT.json',{})
            actual_known,calls,unknown=usage_records(attempt)
            if actual_known!=receipt.get('new_tokens',receipt.get('known_tokens')):raise ValueError('Provider known usage disagrees with execution receipt')
            condition=row.get('condition') or row['variant']
            task={**row,'study':study,'condition':condition,'known_tokens':actual_known,
                'reserved_unknown_tokens':receipt.get('reserved_unknown_tokens',0),'unknown_usage_requests':len(unknown),
                'total_billed_tokens':None if unknown else actual_known,'provider_calls':len(calls),
                'usage_incomplete':receipt.get('usage_incomplete'),'receipt_path':str(attempt/'EXECUTION_RECEIPT.json'),
                'source_tree_sha256':receipt.get('source_tree_sha256'),'profile_sha256':receipt.get('profile_sha256')}
            tasks.append(task)
            summary=read(run/'summarize/result.json',{}).get('payload',{})
            findings={item['candidate_id']:item for item in summary.get('findings',[])}
            report=read(run/'identify/output.json')
            if report is None:
                report=read(attempt/(run.name+'-discovery')/'identify/output.json',{})
            reuse=read(attempt/'FIXED_DISCOVERY_REUSE.json')
            if not report and reuse:report=read(Path(reuse['source'])/'identify/output.json',{})
            dispositions={item['candidate_id']:item for item in read(run/'candidate-validation-dispositions.json',{}).get('candidates',[])}
            for candidate in report.get('candidates',[]):
                cid=candidate['candidate_id'];fate=dispositions.get(cid,{})
                candidate_row={'study':study,'case_id':row['case_id'],'condition':condition,'candidate_id':cid,
                    'audit_status':row['status'],'audit_reason':row['reason'],'cue_name':candidate.get('cue',{}).get('name'),
                    'hypothesis':candidate.get('hypothesis'),'agent_assessment':findings.get(cid,{}).get('assessment'),
                    'canonical_summary_exists':bool(summary),'cheap_state':fate.get('cheap_state'),
                    'construction_status':fate.get('construction',{}).get('status') if isinstance(fate.get('construction'),dict) else fate.get('construction'),
                    'absence_of_shortcut_proven':False,'independent_truth_label':None,'human_adjudication':'PENDING'}
                candidates.append(candidate_row)
                truth.append({key:candidate_row[key] for key in ('study','case_id','condition','candidate_id','cue_name','hypothesis')})
            for path in sorted(run.glob('experiments/*/*.json')):
                record=read(path,{})
                if not record.get('experiment_type'):continue
                base={'study':study,'case_id':row['case_id'],'condition':condition,'candidate_id':record.get('candidate_id'),
                    'audit_status':row['status'],'path':str(path),'sha256':sha(path)}
                tools.append({**base,'experiment_type':record['experiment_type'],'outcome':record.get('outcome'),
                    'metrics_json':json.dumps(record.get('metrics',{}),sort_keys=True),
                    'uncertainty_json':json.dumps(record.get('uncertainty',[]),sort_keys=True)})
                effect=paired_effect(record)
                if effect:effects.append({**base,**effect})
            for path in sorted(run.glob('models/task/*/adaptive_screening.json')):
                report=read(path,{})
                screens.append({'study':study,'case_id':row['case_id'],'condition':condition,'decision':report.get('decision'),
                    'looks':len(report.get('looks',[])),'final_pairs':report.get('looks',[{}])[-1].get('sample_count') if report.get('looks') else None,
                    'path':str(path),'sha256':sha(path),'primary_gate_unchanged':True})
            for path in attempt.glob('feedback_masks/*.json'):
                record=read(path,{})
                masks.append({'case_id':row['case_id'],'condition':condition,'agent':record.get('agent'),
                    'changed_nonempty_text_leaves':changed_text_leaves(record.get('original_payload'),record.get('delivered_payload')),
                    'masked_paths_json':json.dumps(record.get('masked_paths')),'path':str(path),'sha256':sha(path)})
            for path in (attempt/'EXECUTION_RECEIPT.json',run/'summarize/result.json',run/'candidate-validation-dispositions.json'):
                if path.is_file():references.append({'path':str(path),'sha256':sha(path),'kind':'canonical_or_partial'})
    costs=[];missing=[];accounted_invocations=set()
    roots=[MAIN]+[root for root in (MAIN/'extensions').iterdir() if root.is_dir()]
    for root in roots:
        for attempt in sorted((root/'audits').glob('*/*/*/attempt-*')):
            receipt=read(attempt/'EXECUTION_RECEIPT.json')
            if not receipt:continue
            actual_known,calls,unknown=usage_records(attempt,accounted_invocations)
            accounted_invocations.update(record['invocation_id'] for _,record in calls)
            recorded=receipt.get('new_tokens',receipt.get('known_tokens',0))
            if actual_known!=recorded:raise ValueError('Incremental cost mismatch in '+str(attempt))
            costs.append({'attempt':str(attempt),'phase':receipt['phase'],'case_id':receipt['case_id'],'variant':receipt['variant'],
                'known_tokens':recorded,'reserved_unknown_estimate':receipt.get('reserved_unknown_tokens',0),
                'provider_calls':len(calls),'unknown_usage_requests':len(unknown),'total_billed_tokens':None if unknown else recorded,
                'wall_seconds':receipt['wall_seconds'],'receipt_sha256':sha(attempt/'EXECUTION_RECEIPT.json')})
            for path,record in unknown:
                missing.append({'path':str(path),'sha256':sha(path),'run_id':record['run_id'],
                    'runtime_kind':record['runtime_kind'],'model':record['model'],'failure_category':record.get('failure_category'),
                    'actual_tokens':None,'actual_billing':None})
    profile_rows=[]
    for path in sorted((MAIN/'shared_classifiers').glob('*/*/profile.json')):
        member=read(path);cache=path.parent.parent;identity=read(cache/'TRAINING_IDENTITY.json',{})
        weights=path.parent/'best.npz'
        if sha(weights)!=member['weights_sha256']:raise ValueError('Retained training weights changed')
        profile_rows.append({'cache_identity':cache.name,'architecture':member['architecture'],'seed':member['seed'],
            'train_count':len(identity.get('training',[])),'validation_count':len(identity.get('validation',[])),
            'classes':len(member['classes']),'historical_import':(cache/'HISTORICAL_IMPORT.json').exists(),
            'training_validation_checkpoint_wall_seconds':member['elapsed_seconds'],'gpu_kernel_active_seconds':None,
            'weights_sha256':member['weights_sha256'],'weights_path':str(weights)})
    main_tasks=[row for row in tasks if row['study']=='main'];groups=[]
    for condition in ('full','single_agent','fixed_pipeline'):
        group=[row for row in main_tasks if row['condition']==condition]
        groups.append({'condition':condition,'scheduled':len(group),'completed':sum(row['status']=='COMPLETED' for row in group),
            'failed_retained':sum(row['status']=='FAILED_RETAINED' for row in group),'known_tokens':sum(row['known_tokens'] for row in group),
            'reserved_unknown_estimate':sum(row['reserved_unknown_tokens'] for row in group),
            'unknown_usage_requests':sum(row['unknown_usage_requests'] for row in group),
            'wall_minutes':sum(float(row['wall_seconds']) for row in group)/60,
            'interpretation':'Seven dependent conditions from one shared source pool, one Agent audit per arm. Completion is not shortcut efficacy.'})
    for filename,rows,fields in [('AUDIT_STATUS.csv',tasks,['study','case_id','condition','status','reason','known_tokens','reserved_unknown_tokens','unknown_usage_requests','total_billed_tokens','provider_calls','wall_seconds','source_tree_sha256','profile_sha256','receipt_path']),
        ('MAIN_EXECUTION_SUMMARY.csv',groups,list(groups[0])),('CANDIDATE_EVIDENCE.csv',candidates,list(candidates[0])),
        ('TOOL_METRICS.csv',tools,list(tools[0])),('PAIRED_EFFECTS.csv',effects,list(effects[0]) if effects else ['study','case_id','condition','accuracy_drop']),
        ('ADAPTIVE_GATES.csv',screens,list(screens[0])),('FEEDBACK_MANIPULATION.csv',masks,list(masks[0])),
        ('ALL_ATTEMPT_COSTS.csv',costs,list(costs[0])),('MISSING_USAGE.csv',missing,list(missing[0]) if missing else ['path','actual_tokens']),
        ('UNIQUE_CLASSIFIER_PROFILES.csv',profile_rows,list(profile_rows[0]))]:csv_write(tables/filename,rows,fields)
    for row in truth:row.update(rater_1=None,rater_2=None,adjudicated_label=None,status='PENDING')
    csv_write(tables/'HUMAN_ADJUDICATION_PENDING.csv',truth,list(truth[0]))
    pet_local=read(MAIN/'extensions/pet-local-v1/audits/formal/local-01/full/attempt-01/local-01-full/RESULTS.json')
    seeds=[]
    for path in sorted((MAIN/'extensions/seed-v1/audits').glob('*/*/*/attempt-*/*/RESULTS.json')):
        value=read(path);value['source_path']=str(path);value['source_sha256']=sha(path);seeds.append(value)
    write(output/'PET_LOCAL_RESULTS.json',pet_local);write(output/'SEED_REPLICATION_RESULTS.json',seeds)
    specificity_root=MAIN/'extensions/synthetic-specificity-v1/audits/formal/oracle-01/full/attempt-01/oracle-01-full'
    specificity=read(specificity_root/'RESULTS.json')
    if specificity is None:raise ValueError('Final export requires the retained specificity diagnostic outcome')
    write(output/'SYNTHETIC_SPECIFICITY_RESULTS.json',specificity)
    shutil.copyfile(specificity_root/'AUTONOMOUS_PIXEL_SPECIFICITY.csv',tables/'AUTONOMOUS_PIXEL_SPECIFICITY.csv')
    oracle=[]
    for case in specificity['cases']:
        for contrast,value in case['effects'].items():
            oracle.append({'case_id':case['case_id'],'mechanism':case['private_mechanism'],'rho':case['rho'],
                'contrast':contrast,'n_pairs':case['n_pairs'],'accuracy_drop':value['estimate'],
                'lower_95':value['pointwise_95_interval'][0],'upper_95':value['pointwise_95_interval'][1],
                'scope':'Posthoc evaluator oracle; shared source pool; not autonomous discovery or independent confirmation.'})
    csv_write(tables/'EVALUATOR_ORACLE_EFFECTS.csv',oracle,list(oracle[0]) if oracle else ['case_id','contrast'])
    known=sum(row['known_tokens'] for row in costs);reserved=sum(row['reserved_unknown_estimate'] for row in costs)
    if known!=read(MAIN/'MONITOR_SNAPSHOT.json')['cumulative_known_tokens']:raise ValueError('Final known-token ledger disagrees with monitor; investigate, never force a match')
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'agent_audits':len(tasks),'completed':sum(row['status']=='COMPLETED' for row in tasks),
        'failed_retained':sum(row['status']=='FAILED_RETAINED' for row in tasks),'main':groups,
        'known_tokens_all_attempts':known,'reserved_unknown_estimates':reserved,'missing_usage_requests':len(missing),
        'verified_total_billed_tokens':None,'actual_currency_cost':None,'candidate_rows':len(candidates),
        'paired_effect_rows':len(effects),'scope':'Main21 + controlled6 + Pet1; development/continuations accounted separately; diagnostics/extra classifiers are not Agent audits.',
        'independent_synthetic_source_pools':1,'human_adjudication':'PENDING','effectiveness_precision_recall_fpr':None,
        'new_classifier_profiles':sum(not row['historical_import'] for row in profile_rows),'historical_classifier_imports':sum(row['historical_import'] for row in profile_rows),
        'measured_new_profile_wall_seconds':sum(row['training_validation_checkpoint_wall_seconds'] for row in profile_rows if not row['historical_import']),
        'measured_gpu_kernel_active_seconds':None,'pet_local':pet_local,'seed_replication':seeds,
        'synthetic_specificity':specificity}
    write(output/'SUMMARY.json',summary);write(output/'SOURCE_REFERENCES.json',references)
    identity=source_identity(require_clean=False)
    write(output/'reproducibility/EXPORT_SOURCE.json',{'source':identity,'source_archive_sha256':archive_source(output/'reproducibility/export-source.zip',identity['source_tree_sha256'])})
    for name in ('ESWA_single_device_prompt_2026-10-18.md','ESWA_five_day_execution_override_2026-10-18.md'):
        shutil.copyfile(Path('D:/PeiYuan Hao/桌面/ShortcutMiner')/name,output/'reproducibility'/name)
    write(output/'PENDING_AND_DEFERRED.json',{'human_matching_and_fidelity_review':'PENDING; no real human labels received.',
        'discovery_precision_recall_fpr':'UNAVAILABLE pending independent candidate-to-truth matching; never zero-fill.',
        'pet_agent_audit':'FAILED_RETAINED: unknown multimodal usage after timeout; evaluator diagnostics are separate.',
        'pet_text_localization':'DEFERRED: validated localized text masks unavailable.',
        'watermark_extra_seeds':'DEFERRED by original probe expansion gate; no rescue reruns.',
        'patch_ablation_extra_seeds':'NOT_EXECUTED per predeclared budget scope reduction.',
        'rho_0_75_tasks':'NOT_EXECUTED per pre-outcome downgrade, main fixed seven-task scope.',
        'gpu_active_time_and_invoice':'UNAVAILABLE; training wall and known/reserved/unknown requests reported separately.',
        'new_pet_confirmation_pool':'NOT_AVAILABLE; previously inspected historical pool/checkpoint.',
        'extra_datasets_architectures_LLMs':'NOT_EXECUTED per five-day scope.'})
    print(json.dumps({'output':str(output),'audits':len(tasks),'completed':summary['completed'],'failed_retained':summary['failed_retained'],
        'known_tokens':known,'missing_usage_requests':len(missing),'paired_effect_rows':len(effects)}))


if __name__=='__main__':main()
