"""Per-member paired intervention measurements; no verdict or threshold tuning."""
import numpy as np
from shortcutminer.evaluation.calibration import bootstrap_interval

def evaluate_paired_members(model,artifact,original_paths,modified_paths,sample_ids,labels,parameters,*,member_logits=None):
    before,after=member_logits if member_logits is not None else (model.member_logits(original_paths),model.member_logits(modified_paths))
    if set(before)!=set(after): raise ValueError('Paired model membership changed')
    profiles=[];predictions={};intervals=[];labels_array=np.asarray(labels)
    for name in before:
        first=model.classes_[before[name].argmax(1)];second=model.classes_[after[name].argmax(1)]
        correct_first=(first==labels_array).astype(float);correct_second=(second==labels_array).astype(float)
        drop=correct_first-correct_second
        flips=(first!=second).astype(float)
        profiles.append({'profile':name,'pair_count':len(sample_ids),'original_accuracy':float(correct_first.mean()),
            'counterfactual_accuracy':float(correct_second.mean()),'accuracy_drop':float(drop.mean()),
            'prediction_flip_rate':float(flips.mean()),'prediction_flip_count':int(flips.sum())})
        from shortcutminer.experiment.prediction_diagnostics import paired_score_diagnostics
        profiles[-1].update(paired_score_diagnostics(before[name],after[name],labels,model.classes_))
        for metric,values in [('accuracy_drop',drop),('prediction_flip_rate',flips)]:
            interval=bootstrap_interval(metric=f'{name}_{metric}',sample_count=len(sample_ids),
                statistic=lambda idx:float(values[idx].mean()),confidence_level=parameters.confidence_level,
                iterations=parameters.bootstrap_iterations,seed=parameters.seed,resampling_unit='counterfactual_pair')
            intervals.append(interval.model_dump(mode='json'))
        predictions[name]=[{'sample_id':sid,'label':label,'prediction_original':str(a),'prediction_counterfactual':str(b),
            'logits_original':x.tolist(),'logits_counterfactual':y.tolist()}
            for sid,label,a,b,x,y in zip(sample_ids,labels,first,second,before[name],after[name],strict=True)]
    summary={}
    for architecture in sorted({p['profile'].split('-seed')[0] for p in profiles}):
        selected=[p for p in profiles if p['profile'].split('-seed')[0]==architecture]
        summary[architecture]={key:{'mean':float(np.mean([p[key] for p in selected])),
            'std':float(np.std([p[key] for p in selected],ddof=1)) if len(selected)>1 else 0.}
            for key in ['original_accuracy','counterfactual_accuracy','accuracy_drop','prediction_flip_rate']}
    return {'comparison_mode':'paired','model_hash':artifact.model_hash,'profiles':profiles,'summary':summary,
        'predictions':predictions,'uncertainty':intervals}
