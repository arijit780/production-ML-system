from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml_system.features.definitions import FEATURE_NAMES, FEATURE_SET_VERSION


@dataclass
class OnlineFeatureRecord:
    entity_id: str
    feature_set_version: str
    last_updated_at: datetime
    features: dict[str, float | None]

    def to_json(self) -> str:
        return json.dumps(
            {
                "entity_id": self.entity_id,
                "feature_set_version": self.feature_set_version,
                "last_updated_at": self.last_updated_at.isoformat(),
                "features": self.features,
            }
        )

    @classmethod
    def from_json(cls, s: str) -> OnlineFeatureRecord:
        d = json.loads(s)
        raw = d.get("features", {})
        features = {k: raw.get(k) for k in FEATURE_NAMES}
        return cls(
            entity_id=d["entity_id"],
            feature_set_version=d["feature_set_version"],
            last_updated_at=datetime.fromisoformat(d["last_updated_at"]),
            features=features,
        )


class OnlineFeatureStore:
    """
    Low-latency KV projection. SQLite backend for dev/single-node; swap for Redis in production.
    Key: entity_id -> precomputed features + last_updated_at.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS online_features (
                entity_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )"""
        )
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def upsert(self, record: OnlineFeatureRecord) -> None:
        conn = self._conn()
        conn.execute(
            "REPLACE INTO online_features (entity_id, payload) VALUES (?, ?)",
            (record.entity_id, record.to_json()),
        )
        conn.commit()
        conn.close()

    def get(self, entity_id: str) -> OnlineFeatureRecord | None:
        conn = self._conn()
        cur = conn.execute(
            "SELECT payload FROM online_features WHERE entity_id = ?", (entity_id,)
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return OnlineFeatureRecord.from_json(row[0])

    def upsert_from_values(
        self,
        entity_id: str,
        values: dict[str, float | None],
        *,
        feature_set_version: str = FEATURE_SET_VERSION,
    ) -> None:
        rec = OnlineFeatureRecord(
            entity_id=entity_id,
            feature_set_version=feature_set_version,
            last_updated_at=datetime.now(timezone.utc),
            features={k: values.get(k) for k in FEATURE_NAMES},
        )
        self.upsert(rec)
