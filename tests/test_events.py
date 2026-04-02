from datetime import datetime, timezone

from ml_system.events.bronze import BronzeStore
from ml_system.events.schema import RawEvent


def test_bronze_dedupe(tmp_path):
    b = BronzeStore(tmp_path)
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    e = RawEvent(
        event_id="x",
        entity_id="e",
        event_type="click",
        occurred_at=t,
        ingested_at=t,
        payload={},
    )
    b.append([e, e])
    df = b.read_dataframe()
    assert len(df) == 1
