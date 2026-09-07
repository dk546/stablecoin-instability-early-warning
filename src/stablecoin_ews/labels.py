"""Persistent downside events and forward warning labels."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stablecoin_ews.contracts import StudyDesign


class LabelError(ValueError):
    """Raised when event labels cannot be constructed without ambiguity."""


@dataclass(frozen=True)
class EventEpisode:
    """One contiguous persistent downside-price episode."""

    asset_id: str
    onset_utc: pd.Timestamp
    confirmed_at_utc: pd.Timestamp
    end_utc: pd.Timestamp
    observed_hours: int


def _normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"asset_id", "timestamp_utc", "price_observed", "price_usd"}
    missing = required.difference(frame.columns)
    if missing:
        raise LabelError(f"price data are missing columns: {sorted(missing)}")
    output = frame.loc[:, list(required)].copy()
    output["timestamp_utc"] = pd.to_datetime(
        output["timestamp_utc"], utc=True, errors="raise"
    )
    if (output["timestamp_utc"].dt.floor("h") != output["timestamp_utc"]).any():
        raise LabelError("timestamps must be exact UTC hours")
    if output.duplicated(["asset_id", "timestamp_utc"]).any():
        raise LabelError("price data contain duplicate asset-hours")
    if output["price_observed"].dtype != bool:
        raise LabelError("price_observed must be boolean")
    observed_values = pd.to_numeric(
        output.loc[output["price_observed"], "price_usd"], errors="coerce"
    )
    if observed_values.isna().any() or observed_values.le(0).any():
        raise LabelError("observed prices must be positive finite numbers")
    output["price_usd"] = pd.to_numeric(output["price_usd"], errors="coerce")
    return output.sort_values(["asset_id", "timestamp_utc"], kind="mergesort")


def detect_event_episodes(
    frame: pd.DataFrame, design: StudyDesign | None = None
) -> tuple[EventEpisode, ...]:
    """Detect runs of consecutive observed hours below the peg threshold."""

    design = design or StudyDesign()
    prices = _normalize_prices(frame)
    episodes: list[EventEpisode] = []
    one_hour = pd.Timedelta(hours=1)
    for asset_id, group in prices.groupby("asset_id", sort=True):
        run: list[pd.Timestamp] = []
        for row in group.itertuples(index=False):
            timestamp = row.timestamp_utc
            breach = (
                bool(row.price_observed) and row.price_usd < design.event_threshold_usd
            )
            contiguous = not run or timestamp - run[-1] == one_hour
            if breach and contiguous:
                run.append(timestamp)
            else:
                if len(run) >= design.event_duration_hours:
                    episodes.append(
                        EventEpisode(
                            asset_id=str(asset_id),
                            onset_utc=run[0],
                            confirmed_at_utc=run[design.event_duration_hours - 1],
                            end_utc=run[-1],
                            observed_hours=len(run),
                        )
                    )
                run = [timestamp] if breach else []
        if len(run) >= design.event_duration_hours:
            episodes.append(
                EventEpisode(
                    asset_id=str(asset_id),
                    onset_utc=run[0],
                    confirmed_at_utc=run[design.event_duration_hours - 1],
                    end_utc=run[-1],
                    observed_hours=len(run),
                )
            )
    return tuple(episodes)


def build_warning_labels(
    frame: pd.DataFrame, design: StudyDesign | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct observable 6-to-24-hour labels with explicit censoring."""

    design = design or StudyDesign()
    prices = _normalize_prices(frame)
    episodes = detect_event_episodes(prices, design)
    episode_frame = pd.DataFrame(
        [
            {
                "asset_id": episode.asset_id,
                "onset_utc": episode.onset_utc,
                "confirmed_at_utc": episode.confirmed_at_utc,
                "end_utc": episode.end_utc,
                "observed_hours": episode.observed_hours,
            }
            for episode in episodes
        ],
        columns=[
            "asset_id",
            "onset_utc",
            "confirmed_at_utc",
            "end_utc",
            "observed_hours",
        ],
    )
    rows: list[dict[str, object]] = []
    for asset_id, group in prices.groupby("asset_id", sort=True):
        group = group.sort_values("timestamp_utc", kind="mergesort")
        observed_by_time = dict(
            zip(group["timestamp_utc"], group["price_observed"], strict=True)
        )
        maximum = group["timestamp_utc"].max()
        asset_episodes = [
            episode for episode in episodes if episode.asset_id == asset_id
        ]
        for record in group.itertuples(index=False):
            timestamp = record.timestamp_utc
            active = next(
                (
                    episode
                    for episode in asset_episodes
                    if episode.onset_utc <= timestamp <= episode.end_utc
                ),
                None,
            )
            if active is not None:
                status = "active_event_excluded"
                label: int | None = None
                observation_end = max(timestamp, active.confirmed_at_utc)
            else:
                horizon_start = timestamp + pd.Timedelta(
                    hours=design.warning_start_hours
                )
                horizon_end = timestamp + pd.Timedelta(hours=design.warning_end_hours)
                confirmation_end = timestamp + pd.Timedelta(
                    hours=design.confirmation_through_hours
                )
                candidates = [
                    episode
                    for episode in asset_episodes
                    if horizon_start <= episode.onset_utc <= horizon_end
                ]
                if maximum < confirmation_end:
                    status = "endpoint_censored"
                    label = None
                    observation_end = maximum
                else:
                    future_grid = pd.date_range(
                        horizon_start, confirmation_end, freq="h", tz="UTC"
                    )
                    complete = all(
                        observed_by_time.get(hour, False) for hour in future_grid
                    )
                    if not complete:
                        status = "future_gap_censored"
                        label = None
                        observation_end = confirmation_end
                    elif candidates:
                        status = "positive"
                        label = 1
                        observation_end = min(
                            episode.confirmed_at_utc for episode in candidates
                        )
                    else:
                        status = "observable_negative"
                        label = 0
                        observation_end = confirmation_end
            rows.append(
                {
                    "asset_id": str(asset_id),
                    "timestamp_utc": timestamp,
                    "label_status": status,
                    "label": label,
                    "prediction_outcome_eligible": label is not None,
                    "label_observation_end_utc": observation_end,
                }
            )
    labels = pd.DataFrame(rows).sort_values(
        ["asset_id", "timestamp_utc"], kind="mergesort"
    )
    labels["label"] = labels["label"].astype("Int64")
    return labels.reset_index(drop=True), episode_frame.reset_index(drop=True)
