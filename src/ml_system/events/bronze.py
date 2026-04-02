from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml_system.events.schema import RawEvent


class BronzeStore:
    """Append-only Parquet-backed bronze with deduplication by event_id."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "events.parquet"

    def append(self, events: Iterable[RawEvent]) -> int:
        rows = []
        for e in events:
            d = e.model_dump()
            d["payload"] = json.dumps(d["payload"], sort_keys=True)
            rows.append(d)
        if not rows:
            return 0
        new_df = pd.DataFrame(rows)
        for col in ("occurred_at", "ingested_at"):
            new_df[col] = pd.to_datetime(new_df[col], utc=True)
        if self._path.exists():
            old = pd.read_parquet(self._path)
            combined = pd.concat([old, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["event_id"], keep="first")
        else:
            combined = new_df.drop_duplicates(subset=["event_id"], keep="first")
        combined.to_parquet(self._path, index=False)
        return len(combined)

    def read_dataframe(self) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame(
                columns=[
                    "event_id",
                    "entity_id",
                    "event_type",
                    "occurred_at",
                    "ingested_at",
                    "payload",
                ]
            )
        df = pd.read_parquet(self._path)
        for col in ("occurred_at", "ingested_at"):
            df[col] = pd.to_datetime(df[col], utc=True)
        return df

    def events_for_entity_up_to(self, entity_id: str, decision_time: datetime) -> list[RawEvent]:
        """All events for entity with occurred_at <= decision_time, stable order."""
        df = self.read_dataframe()
        if df.empty:
            return []
        dt = pd.Timestamp(decision_time).tz_convert("UTC") if hasattr(decision_time, "tzinfo") else pd.Timestamp(decision_time, tz="UTC")
        mask = (df["entity_id"] == entity_id) & (df["occurred_at"] <= dt)
        sub = df.loc[mask].sort_values(["occurred_at", "event_id"])
        out: list[RawEvent] = []
        for _, row in sub.iterrows():
            pay = row["payload"]
            if isinstance(pay, str):
                payload = json.loads(pay) if pay else {}
            elif isinstance(pay, dict):
                payload = pay
            else:
                payload = {}
            out.append(
                RawEvent(
                    event_id=str(row["event_id"]),
                    entity_id=str(row["entity_id"]),
                    event_type=str(row["event_type"]),
                    occurred_at=row["occurred_at"].to_pydatetime(),
                    ingested_at=row["ingested_at"].to_pydatetime(),
                    payload=payload,
                )
            )
        return out
