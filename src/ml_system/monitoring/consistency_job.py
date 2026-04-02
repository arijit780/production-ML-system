from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_system.events.bronze import BronzeStore
from ml_system.feature_store.online import OnlineFeatureStore
from ml_system.features.definitions import FEATURE_NAMES
from ml_system.model_store.bundle import ModelBundle
from ml_system.monitoring.consistency import ConsistencyReport, run_consistency_check
from ml_system.serving.resolver import FeatureResolver


def build_online_path_frame(
    bronze: BronzeStore,
    online: OnlineFeatureStore,
    bundle: ModelBundle,
    keys: pd.DataFrame,
    *,
    entity_col: str = "entity_id",
    as_of_col: str = "as_of_ts",
) -> pd.DataFrame:
    """Score keys using FeatureResolver + model (mirrors /predict without HTTP)."""
    resolver = FeatureResolver(bronze, online)
    rows: list[dict[str, Any]] = []
    for _, row in keys.iterrows():
        eid = str(row[entity_col])
        ts = row[as_of_col]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        elif not hasattr(ts, "hour"):
            ts = pd.Timestamp(ts).to_pydatetime()
        res = resolver.resolve(eid, ts)
        x = res.vector.ordered_array().reshape(1, -1)
        if np.any(np.isnan(x)):
            proba_pos = float("nan")
            pred = -1
        else:
            pr = bundle.predict_proba(x)
            pred = int(bundle.predict(x)[0])
            proba_pos = float(pr[0, 1]) if pr.shape[1] > 1 else float(pr[0, 0])
        rec: dict[str, Any] = {
            entity_col: eid,
            as_of_col: ts,
            "prediction_online": pred,
            "proba_pos_online": proba_pos,
        }
        for k in FEATURE_NAMES:
            rec[f"feat_{k}"] = res.vector.values.get(k)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_batch_online_consistency(
    batch_parquet: str | Path,
    bronze: BronzeStore,
    online: OnlineFeatureStore,
    bundle: ModelBundle,
    *,
    epsilon_feat: float = 1e-6,
    epsilon_prob: float = 1e-4,
) -> ConsistencyReport:
    batch_df = pd.read_parquet(batch_parquet)
    keys = batch_df[["entity_id", "as_of_ts"]].copy()
    online_df = build_online_path_frame(bronze, online, bundle, keys)
    return run_consistency_check(
        batch_df,
        online_df,
        epsilon_feat=epsilon_feat,
        epsilon_prob=epsilon_prob,
    )


def report_to_dict(r: ConsistencyReport) -> dict[str, Any]:
    return {
        "n_rows": r.n_rows,
        "n_divergent_features": r.n_divergent_features,
        "n_divergent_predictions": r.n_divergent_predictions,
        "max_feature_delta": r.max_feature_delta,
        "max_prediction_delta": r.max_prediction_delta,
        "passed": r.passed,
        "epsilon_feat": r.epsilon_feat,
        "epsilon_prob": r.epsilon_prob,
        "sample_bad_rows": r.sample_bad_rows,
    }
