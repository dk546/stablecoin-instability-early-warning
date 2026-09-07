"""Asset-lifecycle and observed-support utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class LifecycleError(ValueError):
    """Raised when hourly asset support cannot be interpreted safely."""


@dataclass(frozen=True)
class SupportInterval:
    """One contiguous interval of observed hourly prices."""

    asset_id: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    observed_hours: int


def _hourly_market(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"asset_id", "timestamp_utc", "price_observed"}
    missing = required.difference(frame.columns)
    if missing:
        raise LifecycleError(f"market data are missing columns: {sorted(missing)}")
    output = frame.loc[:, sorted(required)].copy()
    output["timestamp_utc"] = pd.to_datetime(
        output["timestamp_utc"], utc=True, errors="raise"
    )
    if (output["timestamp_utc"].dt.floor("h") != output["timestamp_utc"]).any():
        raise LifecycleError("market timestamps must be exact UTC hours")
    if output.duplicated(["asset_id", "timestamp_utc"]).any():
        raise LifecycleError("market data contain duplicate asset-hours")
    if output["price_observed"].dtype != bool:
        raise LifecycleError("price_observed must be boolean")
    return output.sort_values(["asset_id", "timestamp_utc"], kind="mergesort")


def observed_support_intervals(frame: pd.DataFrame) -> tuple[SupportInterval, ...]:
    """Identify contiguous observed-price intervals without filling gaps."""

    market = _hourly_market(frame)
    intervals: list[SupportInterval] = []
    one_hour = pd.Timedelta(hours=1)
    for asset_id, group in market.groupby("asset_id", sort=True):
        observed = group.loc[group["price_observed"], "timestamp_utc"].tolist()
        if not observed:
            continue
        start = previous = observed[0]
        count = 1
        for timestamp in observed[1:]:
            if timestamp - previous == one_hour:
                previous = timestamp
                count += 1
                continue
            intervals.append(SupportInterval(str(asset_id), start, previous, count))
            start = previous = timestamp
            count = 1
        intervals.append(SupportInterval(str(asset_id), start, previous, count))
    return tuple(intervals)


def add_support_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark observations within each asset's first-to-last observed interval."""

    market = _hourly_market(frame)
    bounds = (
        market.loc[market["price_observed"]]
        .groupby("asset_id")["timestamp_utc"]
        .agg(["min", "max"])
    )
    output = frame.copy()
    timestamps = pd.to_datetime(output["timestamp_utc"], utc=True, errors="raise")
    output["within_asset_support"] = False
    for asset_id, row in bounds.iterrows():
        mask = output["asset_id"].eq(asset_id) & timestamps.between(
            row["min"], row["max"], inclusive="both"
        )
        output.loc[mask, "within_asset_support"] = True
    return output
