"""Fixed-stake mixture confidence sequence for signed paired correctness.

Coverage assumes independent protected-content groups with a constant conditional
mean, conditional on a model selected without these observations. This is a
superpopulation interval, not a finite-population without-replacement interval.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

# Frozen generic stakes, not fitted to any dataset's observed effect.
STAKES = np.array([-.49, -.25, -.1, .1, .25, .49])
WEIGHTS = np.array([.2, .2, .1, .1, .2, .2])


def log_e_value(mu, counts):
    factors = 1 + STAKES[:, None] * (np.array([-1., 0., 1.])[None, :] - mu)
    return float(logsumexp(np.log(WEIGHTS) + np.log(factors) @ np.asarray(counts)))


def paired_betting_interval(before, after, labels, *, alpha):
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0,1)')
    first = np.asarray(before) == np.asarray(labels)
    second = np.asarray(after) == np.asarray(labels)
    if first.ndim != 1 or not len(first) or first.shape != second.shape:
        raise ValueError('Matched nonempty predictions required')
    changes = first.astype(int) - second.astype(int)
    counts = [int(np.sum(changes == k)) for k in [-1, 0, 1]]
    mean = float(np.mean(changes))
    def boundary(mu):
        return log_e_value(mu, counts) - np.log(1 / alpha)
    lower = -1. if boundary(-1.) <= 0 else float(brentq(boundary, -1., mean))
    upper = 1. if boundary(1.) <= 0 else float(brentq(boundary, mean, 1.))
    return {'sample_count': len(changes), 'accuracy_drop': mean,
            'lower': lower, 'upper': upper, 'alpha_at_look': alpha,
            'method': 'paired_fixed_stake_mixture_cs_v1',
            'discordance_counts': {'gained': counts[0], 'unchanged': counts[1], 'lost': counts[2]},
            'stakes': STAKES.tolist(), 'mixture_weights': WEIGHTS.tolist(),
            'coverage_scope': 'all_looks_constant_conditional_mean'}
