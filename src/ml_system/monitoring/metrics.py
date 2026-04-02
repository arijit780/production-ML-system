from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


@dataclass
class MetricsLog:
    path: Path

    def append_record(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {**record, "logged_at": datetime.now(timezone.utc).isoformat()}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = []
        with self.path.open() as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)


def append_prediction_metrics(
    path: str | Path,
    *,
    period: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> None:
    log = MetricsLog(Path(path))
    rec = {
        "period": period,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
    }
    if extra:
        rec.update(extra)
    log.append_record(rec)


def prediction_distribution_shift(pred_ref: np.ndarray, pred_cur: np.ndarray, bins: int = 20) -> dict[str, float]:
    ref = pred_ref.astype(float)
    cur = pred_cur.astype(float)
    lo, hi = min(ref.min(), cur.min()), max(ref.max(), cur.max())
    if hi <= lo:
        hi = lo + 1.0
    hist_r, edges = np.histogram(ref, bins=bins, range=(lo, hi), density=True)
    hist_c, _ = np.histogram(cur, bins=bins, range=(lo, hi), density=True)
    tv = float(np.sum(np.abs(hist_r - hist_c)) / 2.0)
    return {"total_variation_approx": tv, "mean_ref": float(np.mean(ref)), "mean_cur": float(np.mean(cur))}
