from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index on two samples.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    qs = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    qs = np.unique(qs)
    if len(qs) < 2:
        return 0.0
    exp_hist, edges = np.histogram(expected, bins=qs)
    act_hist, _ = np.histogram(actual, bins=edges)
    exp_pct = exp_hist / max(len(expected), 1)
    act_pct = act_hist / max(len(actual), 1)
    eps = 1e-6
    exp_pct = np.clip(exp_pct, eps, 1.0)
    act_pct = np.clip(act_pct, eps, 1.0)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic D (simple implementation)."""
    a = np.sort(a[~np.isnan(a)])
    b = np.sort(b[~np.isnan(b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    comb = np.concatenate([a, b])
    cdf_a = np.searchsorted(np.r_[-np.inf, a, np.inf], comb, side="right") - 1
    cdf_b = np.searchsorted(np.r_[-np.inf, b, np.inf], comb, side="right") - 1
    cdf_a = cdf_a / len(a)
    cdf_b = cdf_b / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def summarize_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in feature_cols:
        if c not in reference.columns or c not in current.columns:
            continue
        er = reference[c].to_numpy(dtype=float)
        ec = current[c].to_numpy(dtype=float)
        out[c] = {
            "psi": psi(er, ec),
            "ks": ks_statistic(er, ec),
        }
    return out
