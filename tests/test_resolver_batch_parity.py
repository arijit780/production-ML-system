from datetime import datetime, timedelta, timezone

from ml_system.events.bronze import BronzeStore
from ml_system.events.schema import RawEvent
from ml_system.feature_store.online import OnlineFeatureStore
from ml_system.features.engine import FeatureEngine
from ml_system.serving.resolver import FeatureResolver


def test_resolver_matches_engine_when_online_empty(tmp_path):
    bronze = BronzeStore(tmp_path / "b")
    online = OnlineFeatureStore(tmp_path / "o.db")
    T = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    ev = [
        RawEvent(
            event_id="1",
            entity_id="e",
            event_type="click",
            occurred_at=T - timedelta(hours=1),
            ingested_at=T,
            payload={},
        )
    ]
    bronze.append(ev)
    eng = FeatureEngine().compute("e", T, bronze.events_for_entity_up_to("e", T))
    res = FeatureResolver(bronze, online).resolve("e", T)
    assert res.vector.values == eng.values
