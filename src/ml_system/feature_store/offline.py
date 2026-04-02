from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from ml_system.feature_store.metadata import FeatureStoreMeta
from ml_system.features.definitions import FEATURE_NAMES, FEATURE_SET_ID, FEATURE_SET_VERSION


class OfflineFeatureStore:
    """Immutable-by-version Parquet offline feature tables keyed by (entity_id, as_of_ts)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def table_path(self, job_id: str) -> Path:
        return self.root / f"features_{job_id}.parquet"

    def write_batch(
        self,
        rows: list[dict[str, Any]],
        *,
        materialization_job_id: str | None = None,
    ) -> tuple[Path, FeatureStoreMeta]:
        job = materialization_job_id or str(uuid4())
        df = pd.DataFrame(rows)
        for col in ("as_of_ts",):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        path = self.table_path(job)
        df.to_parquet(path, index=False)
        meta = FeatureStoreMeta(
            feature_set_id=FEATURE_SET_ID,
            feature_set_version=FEATURE_SET_VERSION,
            materialization_job_id=job,
            created_at=datetime.now(timezone.utc),
        )
        meta.write(path.with_suffix(".meta.json"))
        return path, meta

    def read(self, materialization_job_id: str) -> pd.DataFrame:
        path = self.table_path(materialization_job_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)

    @staticmethod
    def row_from_vector(
        entity_id: str,
        as_of_ts: datetime,
        values: dict[str, float | None],
        *,
        job_id: str,
    ) -> dict[str, Any]:
        r: dict[str, Any] = {
            "entity_id": entity_id,
            "as_of_ts": as_of_ts,
            "feature_set_version": FEATURE_SET_VERSION,
            "materialization_job_id": job_id,
        }
        for name in FEATURE_NAMES:
            r[name] = values.get(name)
        return r
