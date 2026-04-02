from datetime import datetime, timedelta, timezone

import numpy as np

from ml_system.events.schema import RawEvent
from ml_system.features.definitions import FEATURE_NAMES
from ml_system.features.engine import FeatureEngine


def _ev(eid: str, entity: str, etype: str, when: datetime, value: float | None = None) -> RawEvent:
    payload = {}
    if value is not None:
        payload["value"] = value
    return RawEvent(
        event_id=eid,
        entity_id=entity,
        event_type=etype,
        occurred_at=when,
        ingested_at=when,
        payload=payload,
    )


def test_no_future_leakage():
    T = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    future = T + timedelta(hours=1)
    events = [
        _ev("1", "e1", "interaction", T - timedelta(days=1), 10.0),
        _ev("2", "e1", "interaction", future, 999.0),
    ]
    fv = FeatureEngine().compute("e1", T, events)
    assert fv.values["rolling_mean_value_7d"] == 10.0


def test_window_excludes_before_window_start():
    T = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    old = T - timedelta(days=8)
    events = [
        _ev("1", "e1", "click", old),
        _ev("2", "e1", "click", T - timedelta(days=1)),
    ]
    fv = FeatureEngine().compute("e1", T, events)
    assert fv.values["count_clicks_7d"] == 1.0


def test_determinism_same_events_order_invariant():
    T = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    a = _ev("a", "e1", "click", T - timedelta(hours=2))
    b = _ev("b", "e1", "click", T - timedelta(hours=1))
    v1 = FeatureEngine().compute("e1", T, [b, a]).ordered_array()
    v2 = FeatureEngine().compute("e1", T, [a, b]).ordered_array()
    np.testing.assert_array_equal(v1, v2)


def test_feature_column_order_matches_definitions():
    assert list(FEATURE_NAMES) == [
        "rolling_mean_value_7d",
        "count_clicks_7d",
        "recency_seconds_since_last_click",
        "count_events_24h",
    ]
