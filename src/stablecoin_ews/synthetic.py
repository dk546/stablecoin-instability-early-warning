"""Deterministic synthetic data for examples and interface checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stablecoin_ews.contracts import CONTEXT_ASSETS, STUDY_ASSETS


def synthetic_hourly_inputs(
    hours: int = 1_200, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate structurally valid market and release-aware macro observations."""

    if hours < 240:
        raise ValueError("at least 240 hours are required for the feature windows")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2022-01-01", periods=hours, freq="h", tz="UTC")
    rows: list[dict[str, object]] = []
    for position, asset in enumerate(STUDY_ASSETS + CONTEXT_ASSETS):
        if asset in STUDY_ASSETS:
            price = 1 + rng.normal(0, 0.00035, hours).cumsum() / 30
            event_start = min(900 + position * 12, hours - 12 + position)
            price[event_start : event_start + 4] = np.array(
                [0.988, 0.987, 0.986, 0.991]
            )
            market_cap = (
                (5 + position * 12) * 1e9 * (1 + rng.normal(0, 0.0007, hours).cumsum())
            )
            volume = (0.4 + position * 0.3) * 1e9 * np.exp(rng.normal(0, 0.12, hours))
        else:
            base = 30_000 if asset == "bitcoin" else 2_000
            price = base * np.exp(rng.normal(0, 0.004, hours).cumsum())
            market_cap = price * (19e6 if asset == "bitcoin" else 120e6)
            volume = market_cap * np.exp(rng.normal(-3.5, 0.15, hours))
        for timestamp, price_value, cap_value, volume_value in zip(
            timestamps, price, market_cap, volume, strict=True
        ):
            rows.append(
                {
                    "asset_id": asset,
                    "timestamp_utc": timestamp,
                    "price_observed": True,
                    "price_effective": float(price_value),
                    "market_cap_effective": float(cap_value),
                    "volume_24h_usd_effective": float(volume_value),
                }
            )
    market = pd.DataFrame(rows)
    release_times = pd.date_range(timestamps.min(), timestamps.max(), freq="24h")
    macro_rows: list[dict[str, object]] = []
    bases = {
        "vix_close": 20.0,
        "us_3m_yield": 2.0,
        "us_10y_yield": 2.8,
        "effective_federal_funds_rate": 1.8,
    }
    for feature, base in bases.items():
        values = base + rng.normal(0, 0.03, len(release_times)).cumsum()
        for timestamp, value in zip(release_times, values, strict=True):
            macro_rows.append(
                {
                    "feature_name": feature,
                    "available_at_utc": timestamp + pd.Timedelta(hours=1),
                    "value": float(value),
                }
            )
    return market, pd.DataFrame(macro_rows)
