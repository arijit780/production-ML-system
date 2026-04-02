from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from ml_system.events.bronze import BronzeStore
from ml_system.feature_store.offline import OfflineFeatureStore
from ml_system.features.definitions import FEATURE_NAMES, FEATURE_SET_VERSION
from ml_system.features.engine import FeatureEngine


def build_offline_snapshot_from_bronze(
    bronze: BronzeStore,
    offline: OfflineFeatureStore,
    *,
    entities: list[str],
    as_of_times: list[datetime],
    materialization_job_id: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """
    Point-in-time feature rows for (entity, as_of_ts) from shared FeatureEngine only.
    """
    job = materialization_job_id or str(uuid4())
    engine = FeatureEngine(FEATURE_SET_VERSION)
    rows: list[dict[str, Any]] = []

    far_future = datetime(9999, 1, 1, tzinfo=timezone.utc)

    for entity_id in entities:
        all_ev = bronze.events_for_entity_up_to(entity_id, far_future)
        for as_of_ts in as_of_times:
            fv = engine.compute(entity_id, as_of_ts, all_ev)
            r = OfflineFeatureStore.row_from_vector(
                entity_id,
                as_of_ts,
                fv.values,
                job_id=job,
            )
            r["warnings"] = ";".join(fv.warnings)
            rows.append(r)

    _, meta = offline.write_batch(rows, materialization_job_id=job)
    df = offline.read(meta.materialization_job_id)
    return meta.materialization_job_id, df


def attach_labels(
    features_df: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    entity_col: str = "entity_id",
    as_of_col: str = "as_of_ts",
) -> pd.DataFrame:
    """
    Join labels computed strictly from post as_of_ts outcomes (caller must enforce I6).
    labels columns: entity_id, as_of_ts, label
    """
    f = features_df.copy()
    l = labels.copy()
    f[as_of_col] = pd.to_datetime(f[as_of_col], utc=True)
    l[as_of_col] = pd.to_datetime(l[as_of_col], utc=True)
    return f.merge(
        l[[entity_col, as_of_col, "label"]],
        on=[entity_col, as_of_col],
        how="inner",
    )


def write_dataset_manifest(
    path: str,
    *,
    dataset_version: str,
    feature_job_id: str,
    feature_set_version: str,
    split_manifest: dict[str, Any],
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    import json
    from pathlib import Path

    doc = {
        "dataset_version": dataset_version,
        "feature_materialization_job_id": feature_job_id,
        "feature_set_version": feature_set_version,
        "split": split_manifest,
        "counts": {"train": n_train, "val": n_val, "test": n_test},
        "feature_columns": list(FEATURE_NAMES),
    }
    Path(path).write_text(json.dumps(doc, indent=2))
