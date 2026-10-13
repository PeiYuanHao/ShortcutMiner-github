"""Registered quantitative training/ID/OOD templates and canonical provenance."""
import json
import platform
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from shortcutminer.protocol.evidence import UncertaintyEstimate
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.task_models.torchvision_matrix import TorchvisionResNetMatrix, FEATURE_VERSION
from shortcutminer.task_models.reference import TaskModelError
from shortcutminer.evaluation.calibration import independent_bootstrap_interval
from shortcutminer.experiment.model_reliance import (
    ModelRelianceEvidence, _split_training_rows, _resolve_valid_rows, _canonical_hash,
    _file_hash, _run_root, _run_output, _load_model, _filter_evaluation_rows,
)


def train_matrix(parameters, access):
    if parameters.training_options is None or parameters.feature_version != FEATURE_VERSION:
        raise TaskModelError('CUDA backend requires frozen training options and matching features')
    # max_eval_samples bounds one evaluation partition, not the union of ID/OOD.
    train,validation,test,generated=_split_training_rows(parameters.model_copy(update={
        'max_eval_samples':100000}),access)
    if generated or validation.empty or test.empty:
        raise TaskModelError('CUDA paper training requires explicit train/validation/held-out partitions')
    train,paths,warnings=_resolve_valid_rows(train,parameters,access)
    validation,val_paths,val_warnings=_resolve_valid_rows(validation,parameters,access)
    if warnings or val_warnings: raise TaskModelError('CUDA training refuses silently excluded images')
    id_column=parameters.sample_id_column
    train_ids=train[id_column].astype(str).tolist(); val_ids=validation[id_column].astype(str).tolist(); test_ids=test[id_column].astype(str).tolist()
    if any(set(first)&set(second) for first,second in [(train_ids,val_ids),(train_ids,test_ids),(val_ids,test_ids)]):
        raise TaskModelError('Frozen training/validation/test identities overlap')
    fingerprint=_canonical_hash({'parameters':parameters.model_dump(mode='json'),
        'dataset_hash':_file_hash(access.dataset_file(parameters.metadata_file)),
        'train':list(zip(train_ids,[_file_hash(p) for p in paths])),
        'validation':list(zip(val_ids,[_file_hash(p) for p in val_paths])),
        **({'adaptive_heldout':[(str(row[id_column]),_file_hash(access.dataset_file(str(row[parameters.sample_path_column]))))
             for _,row in test[test[parameters.split_column].astype(str).isin(parameters.screening_split_values)].iterrows()]}
           if parameters.adaptive_screening is not None else {})})
    directory_reference=f'models/task/{parameters.model_id}'; model_reference=f'{directory_reference}/model.json'
    root=_run_root(access); directory=_run_output(access,f'{directory_reference}/placeholder').parent
    model_path=root/model_reference
    if model_path.exists():
        reused_model,artifact=TorchvisionResNetMatrix.load_safe_artifact(model_path,run_root=root,expected_run_id=access.context.run_id)
        if artifact.training_fingerprint != fingerprint: raise TaskModelError('Existing CUDA training identity changed')
        return ModelRelianceEvidence(outcome='completed',metrics={'model_reused':True,'train_count':len(train),'validation_count':len(validation),'test_count':len(test)},
            summary='Matching current-run CUDA training artifacts reused without importing external baseline results.',
            details={'model_id':artifact.model_id,'model_hash':artifact.model_hash},
            input_references=[parameters.metadata_file],artifact_references=[model_reference,artifact.weights_reference,f'{directory_reference}/training_manifest.json',
                *[p['checkpoint_path'].relative_to(root).as_posix() for p in reused_model.profiles],
                *([f'{directory_reference}/adaptive_screening.json',
                   *[f'{directory_reference}/{item[prefix+"_reference"]}' for item in reused_model.adaptive_report['pixel_evidence'] for prefix in (('changed','control','anchor') if 'anchor_reference' in item else ('changed','control'))]]
                  if reused_model.adaptive_report else [])])
    model=TorchvisionResNetMatrix(parameters.model_id,parameters.training_options)
    probe_ids, callback, shared_cache = [], None, None
    if parameters.adaptive_screening is not None:
        from shortcutminer.experiment.adaptive_screening import prepare_probe
        probe_ids, callback = prepare_probe(parameters, access, test, model, directory)
        # Excludes candidate/transform/probe decisions: only training identities
        # and the full frozen training recipe determine reusable current-run weights.
        shared_identity = _canonical_hash({'dataset_hash':_file_hash(access.dataset_file(parameters.metadata_file)),
            'train':list(zip(train_ids,[_file_hash(p) for p in paths])),
            'validation':list(zip(val_ids,[_file_hash(p) for p in val_paths])),
            'labels':train[parameters.label_column].astype(str).tolist(),
            'validation_labels':validation[parameters.label_column].astype(str).tolist(),
            'options':parameters.training_options.model_dump(mode='json'), 'backend':'torchvision_resnet_matrix_v1'})
        shared_cache = root / 'models/shared_probe' / shared_identity
        shared_cache.resolve().relative_to(root.resolve())
    model.fit_matrix(paths,train[parameters.label_column].astype(str).tolist(),val_paths,validation[parameters.label_column].astype(str).tolist(),directory,fingerprint,
                     probe_callback=callback, shared_cache=shared_cache)
    training={'schema_version':'1.0','run_id':access.context.run_id,'candidate_id':access.request.candidate_id,'task_id':access.request.task_id,
        'training_fingerprint':fingerprint,'train_sample_ids':train_ids,'validation_sample_ids':val_ids,'test_sample_ids':test_ids,
        'dataset_hash':_file_hash(access.dataset_file(parameters.metadata_file)),'split_generated_by_runtime':False,
        'training_options':parameters.training_options.model_dump(mode='json'),'checkpoint_selection':'validation_overall_accuracy_then_earliest_epoch',
        'model_backend':parameters.model_backend,'classes':model.classes_.tolist(),
        'execution_environment':{'torch':str(model.torch.__version__),'cuda_runtime':model.torch.version.cuda,
            'cudnn':model.torch.backends.cudnn.version(),'gpu':model.torch.cuda.get_device_name(),
            'python':platform.python_version(),'platform':platform.platform()}}
    if model.adaptive_report is not None:
        training.update(adaptive_report=model.adaptive_report, adaptive_probe_sample_ids=probe_ids,
                        screening_transformation=parameters.screening_transformation.model_dump(mode='json'))
    artifact=model.save_safe_artifact(run_id=access.context.run_id,run_root=root,model_reference=model_reference,training_fingerprint=fingerprint,training_manifest=training)
    manifest_reference=f'{directory_reference}/training_manifest.json'
    atomic_write_json(root/manifest_reference,training,error_type=TaskModelError)
    profiles=[{k:v for k,v in p.items() if k!='checkpoint_path'} for p in model.profiles]
    shared_references = []
    if shared_cache is not None:
        shared_references = [(shared_cache/f"{p['architecture']}-seed{p['seed']}"/name).relative_to(root).as_posix()
                             for p in model.profiles for name in ('profile.json','best.npz')]
    return ModelRelianceEvidence(outcome='completed',metrics={'model_reused':False,'train_count':len(train),'validation_count':len(validation),
        'test_count':len(test),'class_count':len(model.classes_),'trained_model_count':len(profiles)},
        summary='Quantitative tool executed the frozen CUDA training budget using isolated train/validation partitions; adaptive probes may stop after the first profile.',
        details={'model_id':artifact.model_id,'model_hash':artifact.model_hash,'profiles':profiles,
                 'adaptive_screening':model.adaptive_report,
                 'execution_backend':'torchvision_resnet_matrix_v1'},
        input_references=[parameters.metadata_file],artifact_references=[model_reference,artifact.weights_reference,manifest_reference,
            *[p['checkpoint_path'].relative_to(root).as_posix() for p in model.profiles],
            *[(p['checkpoint_path'].parent/'progress.json').relative_to(root).as_posix() for p in model.profiles],
            *([str((directory/'adaptive_screening.json').relative_to(root)).replace('\\','/'),
               *[f'{directory_reference}/{item[prefix+"_reference"]}' for item in model.adaptive_report['pixel_evidence'] for prefix in (('changed','control','anchor') if 'anchor_reference' in item else ('changed','control'))]]
              if model.adaptive_report else []), *shared_references],limitations=list(artifact.limitations))


def evaluate_dataset_shift(parameters, access):
    model,artifact=_load_model(parameters,access)
    deferred = getattr(model, 'adaptive_report', None) is not None and model.adaptive_report['decision'] != 'expand_models'
    if deferred and not parameters.dataset_shift.pilot_diagnostics_when_deferred:
        return ModelRelianceEvidence(outcome='inconclusive',
            metrics={'final_confirmation_executed':False, 'adaptive_probe_decision':model.adaptive_report['decision']},
            summary='Independent OOD confirmation deferred by the frozen adaptive probe budget decision.',
            details={'comparison_mode':'independent','execution_backend':artifact.backend_version},
            limitations=['No independent OOD accuracy was measured; stopped or deferred probe is not absence of shortcuts.'],
            input_references=[parameters.model_reference])
    shift=parameters.dataset_shift
    id_parameters=parameters.model_copy(update={'split_values':shift.id_split_values,'evaluation_partition':'configured_split'})
    ood_parameters=parameters.model_copy(update={'split_values':shift.ood_split_values,'evaluation_partition':'configured_split'})
    id_rows,id_paths,id_warnings=_resolve_valid_rows(_filter_evaluation_rows(id_parameters,access,artifact),id_parameters,access)
    ood_rows,ood_paths,ood_warnings=_resolve_valid_rows(_filter_evaluation_rows(ood_parameters,access,artifact),ood_parameters,access)
    if id_warnings or ood_warnings or not id_paths or not ood_paths:
        raise TaskModelError('OOD split validation failed; no rows silently dropped')
    ids=set(id_rows[parameters.sample_id_column].astype(str)); ood_ids=set(ood_rows[parameters.sample_id_column].astype(str))
    training=json.loads(access.run_file(parameters.model_reference).read_text())['training']
    if ids & ood_ids or (ids|ood_ids)&set(training['validation_sample_ids']):
        raise TaskModelError('ID/OOD overlap or model-selection validation leakage')
    labels_id=id_rows[parameters.label_column].astype(str).tolist(); labels_ood=ood_rows[parameters.label_column].astype(str).tolist()
    logits_id=model.member_logits(id_paths) if hasattr(model,'member_logits') else {'reference':model.predict_logits(id_paths)}
    logits_ood=model.member_logits(ood_paths) if hasattr(model,'member_logits') else {'reference':model.predict_logits(ood_paths)}
    members=[]; predictions={}; uncertainty=[]
    for name in logits_id:
        pred_id=model.classes_[logits_id[name].argmax(1)].tolist(); pred_ood=model.classes_[logits_ood[name].argmax(1)].tolist()
        correct_id=np.asarray(np.asarray(pred_id)==labels_id,dtype=float); correct_ood=np.asarray(np.asarray(pred_ood)==labels_ood,dtype=float)
        delta=float(correct_id.mean()-correct_ood.mean())
        member={'profile':name,'id_count':len(pred_id),'ood_count':len(pred_ood),'id_accuracy':float(correct_id.mean()),
            'ood_accuracy':float(correct_ood.mean()),'id_to_ood_accuracy_drop':delta,
            'id_balanced_accuracy':float(balanced_accuracy_score(labels_id,pred_id)),
            'ood_balanced_accuracy':float(balanced_accuracy_score(labels_ood,pred_ood))}
        members.append(member)
        interval=independent_bootstrap_interval(metric=f'{name}_accuracy_drop',first_count=len(pred_id),second_count=len(pred_ood),
            statistic=lambda first,second:float(correct_id[first].mean()-correct_ood[second].mean()),
            confidence_level=parameters.confidence_level,iterations=parameters.bootstrap_iterations,seed=parameters.seed)
        uncertainty.append(interval)
        def rows(frame,pred,logits):
            return [{'sample_id':str(row[parameters.sample_id_column]),'label':str(row[parameters.label_column]),'prediction':p,'logits':v.tolist()}
                for (_,row),p,v in zip(frame.iterrows(),pred,logits,strict=True)]
        predictions[name]={'id':rows(id_rows,pred_id,logits_id[name]),'ood':rows(ood_rows,pred_ood,logits_ood[name])}
    summary={}
    for architecture in sorted({m['profile'].split('-seed')[0] for m in members}):
        selected=[m for m in members if m['profile'].split('-seed')[0]==architecture]
        summary[architecture]={key:{'mean':float(np.mean([m[key] for m in selected])),
            'std':float(np.std([m[key] for m in selected],ddof=1)) if len(selected)>1 else 0.} for key in ('id_accuracy','ood_accuracy','id_to_ood_accuracy_drop')}
    identity=_canonical_hash({'parameters':parameters.model_dump(mode='json'),'model_hash':artifact.model_hash,
        'id':[(str(r[parameters.sample_id_column]),_file_hash(p)) for (_,r),p in zip(id_rows.iterrows(),id_paths)],
        'ood':[(str(r[parameters.sample_id_column]),_file_hash(p)) for (_,r),p in zip(ood_rows.iterrows(),ood_paths)]})
    reference=f'models/ood/{access.request.task_id}/{identity[:16]}.json'
    atomic_write_json(_run_output(access,reference),{'schema_version':'1.0','run_id':access.context.run_id,
        'candidate_id':access.request.candidate_id,'task_id':access.request.task_id,'model_hash':artifact.model_hash,
        'comparison_mode':'independent','id_split_values':shift.id_split_values,'ood_split_values':shift.ood_split_values,
        'diagnostic_only':bool(deferred),'final_confirmation_executed':not deferred,
        'profiles':members,'summary':summary,'predictions':predictions,'uncertainty':[u.model_dump(mode='json') for u in uncertainty]},error_type=TaskModelError)
    metrics={'id_count':len(id_rows),'ood_count':len(ood_rows),'sample_count':len(id_rows)+len(ood_rows),'evaluated_model_count':len(members)}
    metrics.update(diagnostic_only=bool(deferred),final_confirmation_executed=not deferred)
    for architecture,values in summary.items():
        for key,stats in values.items():
            metrics[f'{architecture}_{key}_mean']=stats['mean']; metrics[f'{architecture}_{key}_std']=stats['std']
    return ModelRelianceEvidence(outcome='completed',metrics=metrics,summary=('Pilot-only descriptive independent ID/OOD evaluation; does not change the adaptive decision or authorize model expansion.' if deferred else 'Same frozen task models evaluated on independent held-out ID and OOD distributions; architecture-level mean/std retain all seeds.'),
        details={'model_id':artifact.model_id,'model_hash':artifact.model_hash,'comparison_mode':'independent','profiles':members,'summary':summary,
                 'execution_backend':artifact.backend_version},
        limitations=['Independent distribution comparison is not a paired counterfactual or causal proof.','Accuracy drop is measured without forcing a high-impact verdict.',
                     *(['Pilot diagnostic only; one trained profile, no across-seed confirmation and no change to frozen paired gate.'] if deferred else [])],
        uncertainty=uncertainty,input_references=[parameters.model_reference,parameters.metadata_file],artifact_references=[reference])
