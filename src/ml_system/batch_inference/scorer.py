from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ml_system.events.bronze import BronzeStore
from ml_system.features.definitions import FEATURE_NAMES, FEATURE_SET_VERSION
from ml_system.features.engine import FeatureEngine
from ml_system.model_store.bundle import ModelBundle, feature_vector_id, load_bundle


@dataclass
class BatchScoreResult:
    predictions: np.ndarray
    proba: np.ndarray
    feature_matrix: np.ndarray
    lineage: dict[str, Any]


class BatchScorer:
    """Offline scoring using the same FeatureEngine as training (invariant I1, I4)."""

    def __init__(self, bronze: BronzeStore, bundle: ModelBundle) -> None:
        self.bronze = bronze
        self.bundle = bundle
        self._engine = FeatureEngine(FEATURE_SET_VERSION)

    @classmethod
    def from_registry(
        cls,
        bronze: BronzeStore,
        *,
        model_store: str | Path,
        model_version: str,
    ) -> BatchScorer:
        bundle = load_bundle(model_store, model_version)
        return cls(bronze, bundle)

    def score_dataframe(
        self,
        keys: pd.DataFrame,
        *,
        entity_col: str = "entity_id",
        as_of_col: str = "as_of_ts",
    ) -> pd.DataFrame:
        rows = []
        for _, row in keys.iterrows():
            eid = str(row[entity_col])
            ts = row[as_of_col]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            all_ev = self.bronze.events_for_entity_up_to(
                eid,
                ts if isinstance(ts, datetime) else pd.Timestamp(ts).to_pydatetime(),
            )
            fv = self._engine.compute(eid, ts, all_ev)
            x = fv.ordered_array().reshape(1, -1)
            if np.any(np.isnan(x)):
                proba = np.array([[np.nan, np.nan]])
                pred = np.array([-1])
            else:
                proba = self.bundle.predict_proba(x)
                pred = self.bundle.predict(x)
            rows.append(
                {
                    entity_col: eid,
                    as_of_col: ts,
                    "prediction": int(pred[0]),
                    "proba_neg": float(proba[0, 0]) if proba.shape[1] > 1 else float(proba[0, 0]),
                    "proba_pos": float(proba[0, 1]) if proba.shape[1] > 1 else np.nan,
                    "feature_vector_id": feature_vector_id(x.ravel()),
                    **{f"feat_{k}": fv.values.get(k) for k in FEATURE_NAMES},
                }
            )
        return pd.DataFrame(rows)

    def write_parquet(self, df: pd.DataFrame, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        job = str(uuid4())
        p = out / f"batch_predictions_{job}.parquet"
        meta = {
            "batch_job_id": job,
            "model_version": self.bundle.model_version,
            "feature_set_version": self.bundle.feature_set_version,
        }
        df.to_parquet(p, index=False)
        (p.with_suffix(".meta.json")).write_text(
            __import__("json").dumps(meta, indent=2)
        )
        return p
