from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import json


@dataclass
class FeatureStoreMeta:
    feature_set_id: str
    feature_set_version: str
    materialization_job_id: str
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set_id": self.feature_set_id,
            "feature_set_version": self.feature_set_version,
            "materialization_job_id": self.materialization_job_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureStoreMeta:
        return cls(
            feature_set_id=d["feature_set_id"],
            feature_set_version=d["feature_set_version"],
            materialization_job_id=d["materialization_job_id"],
            created_at=datetime.fromisoformat(d["created_at"]),
        )

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read(cls, path: str | Path) -> FeatureStoreMeta:
        return cls.from_dict(json.loads(Path(path).read_text()))
