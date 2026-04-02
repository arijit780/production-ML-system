from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_system.features.definitions import FEATURE_NAMES, FEATURE_SET_ID, FEATURE_SET_VERSION
from ml_system.model_store.bundle import ModelBundle, save_bundle
from ml_system.model_store.registry import ModelRegistry
from ml_system.training.snapshot import write_dataset_manifest
from ml_system.training.split import time_based_split


@dataclass
class MispredictionRow:
    entity_id: str
    as_of_ts: datetime
    y_true: int
    y_pred: int
    proba_positive: float
    top_features: list[tuple[str, float]]


def compute_feature_importance_lr(pipe: Pipeline, feature_names: tuple[str, ...]) -> dict[str, float]:
    clf = pipe.named_steps["clf"]
    coef = clf.coef_.ravel()
    total = np.sum(np.abs(coef)) + 1e-12
    return {n: float(abs(c) / total) for n, c in zip(feature_names, coef, strict=True)}


def misprediction_report(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    importance: dict[str, float],
    *,
    top_k: int = 3,
) -> list[MispredictionRow]:
    rows: list[MispredictionRow] = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            continue
        feats = sorted(importance.items(), key=lambda x: -x[1])[:top_k]
        rows.append(
            MispredictionRow(
                entity_id=str(df.iloc[i]["entity_id"]),
                as_of_ts=df.iloc[i]["as_of_ts"],
                y_true=int(y_true[i]),
                y_pred=int(y_pred[i]),
                proba_positive=float(proba[i, 1] if proba.shape[1] > 1 else proba[i]),
                top_features=feats,
            )
        )
    return rows


class TrainingPipeline:
    def __init__(
        self,
        *,
        model_store_root: str | Path,
        registry_root: str | Path,
    ) -> None:
        self.model_store_root = Path(model_store_root)
        self.registry = ModelRegistry(registry_root)

    def run(
        self,
        labeled_features: pd.DataFrame,
        *,
        train_end: datetime,
        val_end: datetime,
        manifest_out: str | Path,
        dataset_version: str | None = None,
        promote: bool = True,
    ) -> ModelBundle:
        ds_ver = dataset_version or str(uuid4())
        train_df, val_df, test_df, split_m = time_based_split(
            labeled_features, as_of_column="as_of_ts", train_end=train_end, val_end=val_end
        )

        X_train = train_df[list(FEATURE_NAMES)].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy()
        X_val = val_df[list(FEATURE_NAMES)].to_numpy(dtype=float)
        y_val = val_df["label"].to_numpy()
        X_test = test_df[list(FEATURE_NAMES)].to_numpy(dtype=float)
        y_test = test_df["label"].to_numpy()

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=42)),
            ]
        )
        pipe.fit(X_train, y_train)

        y_val_p = pipe.predict(X_val)
        y_test_p = pipe.predict(X_test)
        metrics = {
            "val_accuracy": float(accuracy_score(y_val, y_val_p)),
            "val_precision": float(precision_score(y_val, y_val_p, zero_division=0)),
            "val_recall": float(recall_score(y_val, y_val_p, zero_division=0)),
            "test_accuracy": float(accuracy_score(y_test, y_test_p)),
            "test_precision": float(precision_score(y_test, y_test_p, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_test_p, zero_division=0)),
        }

        model_version = str(uuid4())
        importance = compute_feature_importance_lr(pipe, FEATURE_NAMES)
        bundle = ModelBundle(
            model_version=model_version,
            feature_set_id=FEATURE_SET_ID,
            feature_set_version=FEATURE_SET_VERSION,
            training_dataset_version=ds_ver,
            feature_names=FEATURE_NAMES,
            sklearn_pipeline=pipe,
            metrics=metrics,
            feature_importance=importance,
        )
        save_bundle(self.model_store_root, bundle)
        self.registry.register(model_version, promote_production=promote)

        fj = "unknown"
        if "materialization_job_id" in labeled_features.columns and len(labeled_features):
            fj = str(labeled_features.iloc[0]["materialization_job_id"])
        write_dataset_manifest(
            manifest_out,
            dataset_version=ds_ver,
            feature_job_id=fj,
            feature_set_version=FEATURE_SET_VERSION,
            split_manifest=split_m.to_dict(),
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
        )
        return bundle
