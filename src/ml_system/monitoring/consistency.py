from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml_system.features.definitions import FEATURE_NAMES


@dataclass
class ConsistencyReport:
    n_rows: int
    n_divergent_features: int
    n_divergent_predictions: int
    max_feature_delta: float
    max_prediction_delta: float
    sample_bad_rows: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True
    epsilon_feat: float = 1e-6
    epsilon_prob: float = 1e-4


def run_consistency_check(
    batch_df: pd.DataFrame,
    online_df: pd.DataFrame,
    *,
    key_cols: tuple[str, str] = ("entity_id", "as_of_ts"),
    prob_batch: str = "proba_pos",
    prob_online: str = "proba_pos_online",
    epsilon_feat: float = 1e-6,
    epsilon_prob: float = 1e-4,
    max_samples: int = 20,
) -> ConsistencyReport:
    """
    Compare batch vs online paths on aligned keys.
    batch_df must include feat_* columns from BatchScorer; online_df same + prob column.
    """
    m = batch_df.merge(online_df, on=list(key_cols), how="inner", suffixes=("_b", "_o"))
    if m.empty:
        return ConsistencyReport(
            n_rows=0,
            n_divergent_features=0,
            n_divergent_predictions=0,
            max_feature_delta=0.0,
            max_prediction_delta=0.0,
            passed=False,
            epsilon_feat=epsilon_feat,
            epsilon_prob=epsilon_prob,
        )

    bad_feat = 0
    bad_pred = 0
    max_fd = 0.0
    max_pd = 0.0
    samples: list[dict[str, Any]] = []

    for _, row in m.iterrows():
        row_bad = False
        for k in FEATURE_NAMES:
            fb = row.get(f"feat_{k}_b", row.get(f"feat_{k}"))
            fo = row.get(f"feat_{k}_o", row.get(f"feat_{k}"))
            if fb is None or fo is None:
                continue
            if np.isnan(fb) and np.isnan(fo):
                continue
            d = abs(float(fb) - float(fo))
            max_fd = max(max_fd, d)
            if d > epsilon_feat:
                row_bad = True
        if row_bad:
            bad_feat += 1

        pb = row.get(prob_batch)
        po = row.get(prob_online)
        if pb is not None and po is not None and not (np.isnan(pb) and np.isnan(po)):
            pdel = abs(float(pb) - float(po))
            max_pd = max(max_pd, pdel)
            if pdel > epsilon_prob:
                bad_pred += 1
                row_bad = True

        if row_bad and len(samples) < max_samples:
            samples.append({c: row[c] for c in key_cols})

    n = len(m)
    passed = bad_feat == 0 and bad_pred == 0
    return ConsistencyReport(
        n_rows=n,
        n_divergent_features=bad_feat,
        n_divergent_predictions=bad_pred,
        max_feature_delta=max_fd,
        max_prediction_delta=max_pd,
        sample_bad_rows=samples,
        passed=passed,
        epsilon_feat=epsilon_feat,
        epsilon_prob=epsilon_prob,
    )
