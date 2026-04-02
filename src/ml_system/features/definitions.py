from __future__ import annotations

import hashlib
import json
from typing import Any

FEATURE_SET_ID = "default_entity_features"
FEATURE_SET_VERSION = "1.0.0"

# Ordered training / inference column order (invariant I4)
FEATURE_NAMES: tuple[str, ...] = (
    "rolling_mean_value_7d",
    "count_clicks_7d",
    "recency_seconds_since_last_click",
    "count_events_24h",
)


def feature_manifest() -> dict[str, Any]:
    """Versioned manifest for lineage and API validation."""
    body = {
        "feature_set_id": FEATURE_SET_ID,
        "feature_set_version": FEATURE_SET_VERSION,
        "features": [
            {
                "name": "rolling_mean_value_7d",
                "dtype": "float64",
                "window_spec": "7d mean of payload.value for event_type=interaction",
            },
            {
                "name": "count_clicks_7d",
                "dtype": "float64",
                "window_spec": "count event_type=click in 7d",
            },
            {
                "name": "recency_seconds_since_last_click",
                "dtype": "float64",
                "window_spec": "seconds from decision_time to last click occurred_at",
            },
            {
                "name": "count_events_24h",
                "dtype": "float64",
                "window_spec": "count all event types in 24h",
            },
        ],
    }
    h = hashlib.sha256(
        json.dumps(body["features"], sort_keys=True).encode()
    ).hexdigest()[:12]
    body["determinism_hash"] = h
    return body
