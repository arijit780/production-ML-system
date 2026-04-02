from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_system.events.bronze import BronzeStore
from ml_system.feature_store.online import OnlineFeatureStore
from ml_system.model_store.bundle import feature_vector_id, load_bundle
from ml_system.model_store.registry import ModelRegistry
from ml_system.serving.resolver import FeatureResolver


class PredictRequest(BaseModel):
    entity_id: str
    decision_time: datetime
    model_version: str | None = None
    feature_overrides: dict[str, float] | None = Field(default=None, description="Debug only")


class PredictResponse(BaseModel):
    prediction: int
    scores: list[float]
    model_version: str
    feature_set_version: str
    feature_vector_id: str
    warnings: list[str]


def create_app(
    *,
    bronze_path: str | Path,
    online_db: str | Path,
    model_store: str | Path,
    registry_path: str | Path,
    fail_on_feature_version_mismatch: bool = True,
) -> FastAPI:
    bronze = BronzeStore(bronze_path)
    online = OnlineFeatureStore(online_db)
    registry = ModelRegistry(registry_path)
    resolver = FeatureResolver(bronze, online)

    app = FastAPI(title="ML System Inference")

    def _get_bundle(version: str | None):
        mv = version or registry.get_production_version()
        if not mv:
            raise HTTPException(503, "no production model")
        return load_bundle(model_store, mv), mv

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> Any:
        bundle, mv = _get_bundle(req.model_version)
        if bundle.feature_set_version != resolver.feature_set_version:
            if fail_on_feature_version_mismatch:
                raise HTTPException(
                    409,
                    detail="model feature_set_version incompatible with serving resolver",
                )

        res = resolver.resolve(req.entity_id, req.decision_time)
        x = res.vector.ordered_array().reshape(1, -1)

        if req.feature_overrides:
            cols = list(bundle.feature_names)
            for k, v in req.feature_overrides.items():
                if k not in cols:
                    continue
                idx = cols.index(k)
                x[0, idx] = v

        if np.any(np.isnan(x)):
            raise HTTPException(
                400,
                detail="missing required features after resolution; partial_feature_vector not supported for this model",
            )

        proba = bundle.predict_proba(x)
        pred = bundle.predict(x)
        fid = feature_vector_id(x.ravel())
        warnings = list(res.vector.warnings)

        return PredictResponse(
            prediction=int(pred[0]),
            scores=[float(proba[0, 0]), float(proba[0, 1])] if proba.shape[1] >= 2 else [float(proba[0, 0])],
            model_version=mv,
            feature_set_version=bundle.feature_set_version,
            feature_vector_id=fid,
            warnings=warnings,
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
