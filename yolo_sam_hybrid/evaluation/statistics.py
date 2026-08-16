"""Statistical summaries for case-level or image-level metrics."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0


def bootstrap_ci(values: Iterable[float], n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 17821) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(np.mean(sample))
    alpha = 1 - ci
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_test(values_a: Iterable[float], values_b: Iterable[float]) -> dict:
    a = np.asarray(list(values_a), dtype=np.float64)
    b = np.asarray(list(values_b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have identical shapes")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return {"test": "insufficient_pairs", "p_value": float("nan")}
    try:
        from scipy.stats import shapiro, ttest_rel, wilcoxon

        diff = a - b
        normal_p = shapiro(diff).pvalue if diff.size >= 3 else 0.0
        if normal_p >= 0.05:
            stat, p = ttest_rel(a, b)
            return {"test": "paired_t_test", "statistic": float(stat), "p_value": float(p), "normality_p": float(normal_p)}
        stat, p = wilcoxon(a, b)
        return {"test": "wilcoxon_signed_rank", "statistic": float(stat), "p_value": float(p), "normality_p": float(normal_p)}
    except Exception as exc:
        return {"test": "failed", "error": str(exc), "p_value": float("nan")}
