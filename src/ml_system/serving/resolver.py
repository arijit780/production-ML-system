from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ml_system.events.bronze import BronzeStore
from ml_system.feature_store.online import OnlineFeatureStore
from ml_system.features.definitions import FEATURE_NAMES, FEATURE_SET_VERSION
from ml_system.features.engine import FeatureEngine, FeatureVector


@dataclass
class FeatureResolution:
    """Outcome of precomputed + on-demand merge (single engine for on-demand)."""

    vector: FeatureVector
    used_online_keys: set[str] = field(default_factory=set)
    recomputed_full: bool = False

    @property
    def warnings(self) -> list[str]:
        return list(self.vector.warnings)


class FeatureResolver:
    """
    1) Load online projection when version matches.
    2) Ground-truth from FeatureEngine on Bronze up to decision_time.
    3) For each feature: prefer engine value if online missing/stale/version mismatch.
    """

    def __init__(
        self,
        bronze: BronzeStore,
        online: OnlineFeatureStore,
        *,
        feature_set_version: str = FEATURE_SET_VERSION,
        stale_seconds: float = 86400.0,
    ) -> None:
        self.bronze = bronze
        self.online = online
        self.feature_set_version = feature_set_version
        self.stale_seconds = stale_seconds
        self._engine = FeatureEngine(feature_set_version)

    def resolve(self, entity_id: str, decision_time: datetime) -> FeatureResolution:
        T = decision_time
        if T.tzinfo is None:
            T = T.replace(tzinfo=timezone.utc)

        events = self.bronze.events_for_entity_up_to(entity_id, T)
        full = self._engine.compute(entity_id, T, events)
        warnings = list(full.warnings)
        used_online: set[str] = set()
        merged = dict(full.values)
        rec = self.online.get(entity_id)
        now = datetime.now(timezone.utc)

        version_ok = rec is not None and rec.feature_set_version == self.feature_set_version
        stale = False
        if rec is not None:
            lu = rec.last_updated_at
            if lu.tzinfo is None:
                lu = lu.replace(tzinfo=timezone.utc)
            else:
                lu = lu.astimezone(timezone.utc)
            age = (now - lu).total_seconds()
            if age > self.stale_seconds:
                stale = True
                warnings.append("stale_online_projection")

        if not version_ok:
            if rec is not None:
                warnings.append("online_feature_set_version_mismatch")
            return FeatureResolution(
                vector=FeatureVector(
                    entity_id=entity_id,
                    decision_time=decision_time,
                    feature_set_version=self.feature_set_version,
                    values=merged,
                    warnings=warnings,
                ),
                used_online_keys=set(),
                recomputed_full=True,
            )

        if stale:
            return FeatureResolution(
                vector=FeatureVector(
                    entity_id=entity_id,
                    decision_time=decision_time,
                    feature_set_version=self.feature_set_version,
                    values=merged,
                    warnings=warnings,
                ),
                used_online_keys=set(),
                recomputed_full=True,
            )

        # Fill only where engine produced None; never overwrite non-null engine values (I1 skew guard).
        for k in FEATURE_NAMES:
            ov = rec.features.get(k) if rec else None
            if merged.get(k) is None and ov is not None:
                merged[k] = ov
                used_online.add(k)

        if any(merged.get(k) is None for k in FEATURE_NAMES):
            warnings.append("partial_feature_vector")

        return FeatureResolution(
            vector=FeatureVector(
                entity_id=entity_id,
                decision_time=decision_time,
                feature_set_version=self.feature_set_version,
                values=merged,
                warnings=warnings,
            ),
            used_online_keys=used_online,
            recomputed_full=False,
        )
