"""Paired uncertainty estimators that preserve temporal and asset structure."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Metric = Callable[[np.ndarray, np.ndarray], float]


def _moving_block_positions(
    frame: pd.DataFrame, rng: np.random.Generator, block_hours: int
) -> np.ndarray:
    sampled: list[int] = []
    for _, group in frame.groupby("asset_id", sort=True):
        positions = group.index.to_numpy(dtype=int)
        if not len(positions):
            continue
        draws: list[int] = []
        while len(draws) < len(positions):
            start = int(rng.integers(0, len(positions)))
            block = positions[start : min(len(positions), start + block_hours)]
            if not len(block):
                block = positions[-1:]
            draws.extend(block.tolist())
        sampled.extend(draws[: len(positions)])
    return np.asarray(sampled, dtype=int)


def paired_moving_block_bootstrap(
    frame: pd.DataFrame,
    m0_probability: str,
    m1_probability: str,
    metric: Metric,
    *,
    replicates: int = 2_000,
    block_hours: int = 168,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap a paired M1-minus-M0 row metric within each asset."""

    required = {"asset_id", "timestamp_utc", "label", m0_probability, m1_probability}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"bootstrap data are missing columns: {sorted(missing)}")
    if replicates < 1 or block_hours < 1:
        raise ValueError("bootstrap settings must be positive")
    ordered = frame.sort_values(
        ["asset_id", "timestamp_utc"], kind="mergesort"
    ).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replicate in range(replicates):
        positions = _moving_block_positions(ordered, rng, block_hours)
        labels = ordered.loc[positions, "label"].to_numpy(dtype=int)
        m0_value = float(
            metric(labels, ordered.loc[positions, m0_probability].to_numpy(dtype=float))
        )
        m1_value = float(
            metric(labels, ordered.loc[positions, m1_probability].to_numpy(dtype=float))
        )
        rows.append(
            {
                "replicate": replicate,
                "m0": m0_value,
                "m1": m1_value,
                "m1_minus_m0": m1_value - m0_value,
            }
        )
    return pd.DataFrame(rows)


def percentile_interval(
    values: object, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a finite percentile interval for bootstrap replicates."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array) or not 0 < confidence < 1:
        raise ValueError("confidence interval input is invalid")
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(array, [tail, 1 - tail], method="linear")
    return float(lower), float(upper)
