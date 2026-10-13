"""Descriptive paired-score diagnostics; no promotion or hypothesis-test decision."""
import numpy as np


def paired_score_diagnostics(original, changed, labels, classes):
    first=np.asarray(original,dtype=float);second=np.asarray(changed,dtype=float)
    names=[str(c) for c in classes];targets=[str(y) for y in labels]
    if first.ndim!=2 or first.shape!=second.shape or first.shape!=(len(targets),len(names)) or len(names)<2 or not targets:
        raise ValueError('Aligned nonempty paired logits, labels and classes required')
    if len(set(names))!=len(names) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Unique classes and finite logits required')
    positions=np.asarray([names.index(y) for y in targets]);rows=np.arange(len(targets))
    def margin(values):
        other=values.copy();other[rows,positions]=-np.inf
        return values[rows,positions]-other.max(axis=1)
    a=margin(first);b=margin(second)
    return {'original_true_class_margin_mean':float(a.mean()),'counterfactual_true_class_margin_mean':float(b.mean()),
            'logit_margin_delta':float((b-a).mean()),'true_class_margin_decreased_fraction':float(np.mean(b<a))}
