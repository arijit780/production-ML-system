from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class TimeSplitManifest:
    train_end: datetime
    val_end: datetime
    as_of_column: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_end": self.train_end.isoformat(),
            "val_end": self.val_end.isoformat(),
            "as_of_column": self.as_of_column,
        }


def time_based_split(
    df: pd.DataFrame,
    *,
    as_of_column: str = "as_of_ts",
    train_end: datetime,
    val_end: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TimeSplitManifest]:
    """
    Contiguous time split (invariant I3: no random split).
    train: as_of < train_end
    val: train_end <= as_of < val_end
    test: as_of >= val_end
    """
    ts = pd.to_datetime(df[as_of_column], utc=True)
    tr = pd.Timestamp(train_end)
    if tr.tzinfo is None:
        tr = tr.tz_localize("UTC")
    else:
        tr = tr.tz_convert("UTC")
    ve = pd.Timestamp(val_end)
    if ve.tzinfo is None:
        ve = ve.tz_localize("UTC")
    else:
        ve = ve.tz_convert("UTC")

    train_df = df.loc[ts < tr].copy()
    val_df = df.loc[(ts >= tr) & (ts < ve)].copy()
    test_df = df.loc[ts >= ve].copy()
    manifest = TimeSplitManifest(train_end=train_end, val_end=val_end, as_of_column=as_of_column)
    return train_df, val_df, test_df, manifest
