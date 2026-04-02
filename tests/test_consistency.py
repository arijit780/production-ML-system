import numpy as np
import pandas as pd

from ml_system.monitoring.consistency import run_consistency_check


def test_consistency_passes_identical_frames():
    batch = pd.DataFrame(
        {
            "entity_id": ["a"],
            "as_of_ts": [pd.Timestamp("2024-01-01", tz="UTC")],
            "feat_rolling_mean_value_7d": [1.0],
            "feat_count_clicks_7d": [2.0],
            "feat_recency_seconds_since_last_click": [3.0],
            "feat_count_events_24h": [4.0],
            "proba_pos": [0.7],
        }
    )
    online = batch.rename(columns={"proba_pos": "proba_pos_online"})
    r = run_consistency_check(batch, online, prob_online="proba_pos_online")
    assert r.passed
