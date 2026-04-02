from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import numpy as np

from ml_system.events.schema import RawEvent
from ml_system.features.definitions import FEATURE_NAMES, FEATURE_SET_VERSION


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_value(ev: RawEvent) -> float:
    v = ev.payload.get("value") if isinstance(ev.payload, Mapping) else None
    if v is None:
        return 0.0
    return float(v)


@dataclass
class FeatureVector:
    """Resolved features in training column order."""

    entity_id: str
    decision_time: datetime
    feature_set_version: str
    values: dict[str, float | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def ordered_array(self) -> np.ndarray:
        arr = []
        for name in FEATURE_NAMES:
            v = self.values.get(name)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                arr.append(np.nan)
            else:
                arr.append(float(v))
        return np.asarray(arr, dtype=np.float64)

    def ordered_dict(self) -> dict[str, float | None]:
        return {k: self.values.get(k) for k in FEATURE_NAMES}


class FeatureEngine:
    """
    Single implementation for training, batch, and online (invariant I1).
    Windows: include events with occurred_at in (decision_time - window, decision_time] (closed at T).
    """

    def __init__(self, feature_set_version: str = FEATURE_SET_VERSION) -> None:
        self.feature_set_version = feature_set_version

    def _filter_pit(self, events: list[RawEvent], decision_time: datetime) -> list[RawEvent]:
        T = _as_utc(decision_time)
        out = [e for e in events if _as_utc(e.occurred_at) <= T]
        out.sort(key=lambda e: (_as_utc(e.occurred_at), e.event_id))
        return out

    def _in_window(self, ev: RawEvent, decision_time: datetime, window: timedelta) -> bool:
        T = _as_utc(decision_time)
        t = _as_utc(ev.occurred_at)
        return (T - window) < t <= T

    def compute(
        self,
        entity_id: str,
        decision_time: datetime,
        events: list[RawEvent],
    ) -> FeatureVector:
        pit = self._filter_pit(events, decision_time)
        T = _as_utc(decision_time)
        w7 = timedelta(days=7)
        w24 = timedelta(hours=24)

        vals: dict[str, float | None] = {}
        warnings: list[str] = []

        # rolling_mean_value_7d: interaction events only
        inter = [
            e
            for e in pit
            if e.event_type == "interaction" and self._in_window(e, decision_time, w7)
        ]
        if inter:
            vals["rolling_mean_value_7d"] = float(np.mean([_event_value(e) for e in inter]))
        else:
            vals["rolling_mean_value_7d"] = None
            warnings.append("no_interactions_in_7d_window")

        clicks_7d = [e for e in pit if e.event_type == "click" and self._in_window(e, decision_time, w7)]
        vals["count_clicks_7d"] = float(len(clicks_7d))

        clicks_all = [e for e in pit if e.event_type == "click"]
        if clicks_all:
            last = max(clicks_all, key=lambda e: (_as_utc(e.occurred_at), e.event_id))
            delta = T - _as_utc(last.occurred_at)
            vals["recency_seconds_since_last_click"] = float(delta.total_seconds())
        else:
            vals["recency_seconds_since_last_click"] = None
            warnings.append("no_click_history")

        recent = [e for e in pit if self._in_window(e, decision_time, w24)]
        vals["count_events_24h"] = float(len(recent))

        return FeatureVector(
            entity_id=entity_id,
            decision_time=decision_time,
            feature_set_version=self.feature_set_version,
            values=vals,
            warnings=warnings,
        )


def compute_feature_matrix_row(
    entity_id: str,
    decision_time: datetime,
    events: list[RawEvent],
    feature_set_version: str = FEATURE_SET_VERSION,
) -> np.ndarray:
    """Convenience: same as FeatureEngine.compute().ordered_array()."""
    return FeatureEngine(feature_set_version).compute(entity_id, decision_time, events).ordered_array()
