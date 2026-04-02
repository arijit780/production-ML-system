from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from ml_system.features.definitions import FEATURE_NAMES, FEATURE_SET_ID, FEATURE_SET_VERSION


@dataclass
class ModelBundle:
    model_version: str
    feature_set_id: str
    feature_set_version: str
    training_dataset_version: str
    feature_names: tuple[str, ...]
    sklearn_pipeline: Pipeline
    metrics: dict[str, float]
    feature_importance: dict[str, float]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.sklearn_pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.sklearn_pipeline.predict(X)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "feature_set_id": self.feature_set_id,
            "feature_set_version": self.feature_set_version,
            "training_dataset_version": self.training_dataset_version,
            "feature_names": list(self.feature_names),
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }


def save_bundle(root: str | Path, bundle: ModelBundle) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    mid = bundle.model_version
    d = root / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(
        json.dumps(bundle.to_metadata_dict(), indent=2)
    )
    with (d / "model.pkl").open("wb") as f:
        pickle.dump(bundle.sklearn_pipeline, f)
    return d


def load_bundle(root: str | Path, model_version: str) -> ModelBundle:
    d = Path(root) / model_version
    meta = json.loads((d / "metadata.json").read_text())
    with (d / "model.pkl").open("rb") as f:
        pipe = pickle.load(f)
    return ModelBundle(
        model_version=meta["model_version"],
        feature_set_id=meta["feature_set_id"],
        feature_set_version=meta["feature_set_version"],
        training_dataset_version=meta["training_dataset_version"],
        feature_names=tuple(meta["feature_names"]),
        sklearn_pipeline=pipe,
        metrics=meta.get("metrics", {}),
        feature_importance=meta.get("feature_importance", {}),
    )


def feature_vector_id(values: np.ndarray) -> str:
    s = np.nan_to_num(values, nan=-9999.0).tobytes()
    return hashlib.sha256(s).hexdigest()[:16]


def build_default_bundle_stub() -> ModelBundle:
    """Placeholder types for validation; training overwrites."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    return ModelBundle(
        model_version="unset",
        feature_set_id=FEATURE_SET_ID,
        feature_set_version=FEATURE_SET_VERSION,
        training_dataset_version="unset",
        feature_names=FEATURE_NAMES,
        sklearn_pipeline=pipe,
        metrics={},
        feature_importance={},
    )
