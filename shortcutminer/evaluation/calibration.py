"""Deterministic statistical calibration helpers without decision thresholds."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from shortcutminer.protocol.evidence import UncertaintyEstimate


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return deterministic, monotone BH-FDR adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("p-values must be a finite one-dimensional sequence")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be bounded by zero and one")
    count = len(values)
    if count == 0:
        return []
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = ranked * count / np.arange(1, count + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def bootstrap_interval(
    *,
    metric: str,
    sample_count: int,
    statistic: Callable[[np.ndarray], float | None],
    confidence_level: float,
    iterations: int,
    seed: int,
    resampling_unit: str,
) -> UncertaintyEstimate:
    """Percentile bootstrap over row indices with explicit unavailable states."""
    if iterations <= 0:
        return UncertaintyEstimate(
            metric=metric,
            available=False,
            reason_unavailable="Bootstrap was disabled by configuration.",
        )
    if sample_count < 2:
        return UncertaintyEstimate(
            metric=metric,
            available=False,
            reason_unavailable="At least two resampling units are required.",
        )
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        value = statistic(rng.integers(0, sample_count, size=sample_count))
        if value is not None and np.isfinite(value):
            estimates.append(float(value))
    if len(estimates) < max(2, iterations // 2):
        return UncertaintyEstimate(
            metric=metric,
            available=False,
            reason_unavailable="Too few valid bootstrap replicates were estimable.",
        )
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return UncertaintyEstimate(
        metric=metric,
        method="percentile_bootstrap",
        confidence_level=confidence_level,
        lower=float(lower),
        upper=float(upper),
        standard_error=float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
        bootstrap_iterations=iterations,
        seed=seed,
        resampling_unit=resampling_unit,
        available=True,
    )


__all__ = ["benjamini_hochberg", "bootstrap_interval"]
