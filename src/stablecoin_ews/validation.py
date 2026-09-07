"""Common forward-only expanding-window validation plans."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stablecoin_ews.contracts import StudyDesign


class ValidationError(ValueError):
    """Raised when a forward-only fold plan cannot be formed."""


@dataclass(frozen=True)
class Fold:
    """One daily out-of-sample block and its admissible training population."""

    index: int
    test_start_utc: pd.Timestamp
    test_end_exclusive_utc: pd.Timestamp
    training_index: tuple[int, ...]
    test_index: tuple[int, ...]


def model_population(
    labels: pd.DataFrame, features: pd.DataFrame, design: StudyDesign | None = None
) -> pd.DataFrame:
    """Join labels and features and retain only fully observable rows."""

    design = design or StudyDesign()
    keys = ["asset_id", "timestamp_utc"]
    if labels.duplicated(keys).any() or features.duplicated(keys).any():
        raise ValidationError("labels and features must have unique asset-hour keys")
    joined = labels.merge(features, on=keys, how="inner", validate="one_to_one")
    required = {
        "label",
        "prediction_outcome_eligible",
        "label_observation_end_utc",
        "feature_eligibility",
    }
    missing = required.difference(joined.columns)
    if missing:
        raise ValidationError(f"model inputs are missing columns: {sorted(missing)}")
    eligible = (
        joined["prediction_outcome_eligible"].astype(bool)
        & joined["feature_eligibility"].astype(bool)
        & joined["label"].notna()
    )
    output = joined.loc[eligible].copy()
    output["timestamp_utc"] = pd.to_datetime(
        output["timestamp_utc"], utc=True, errors="raise"
    )
    output["label_observation_end_utc"] = pd.to_datetime(
        output["label_observation_end_utc"], utc=True, errors="raise"
    )
    output["label"] = output["label"].astype(int)
    if (output["label_observation_end_utc"] < output["timestamp_utc"]).any():
        raise ValidationError("outcome availability precedes its prediction timestamp")
    output["admissible_at_utc"] = output["label_observation_end_utc"] + pd.Timedelta(
        hours=design.training_embargo_hours
    )
    return output.sort_values(["timestamp_utc", "asset_id"], kind="mergesort")


def expanding_daily_folds(
    population: pd.DataFrame, design: StudyDesign | None = None
) -> tuple[Fold, ...]:
    """Create common daily tests whose training rows are causally admissible."""

    design = design or StudyDesign()
    required = {"timestamp_utc", "admissible_at_utc", "label", "asset_id"}
    missing = required.difference(population.columns)
    if missing:
        raise ValidationError(f"population is missing columns: {sorted(missing)}")
    if population.empty:
        raise ValidationError("model population is empty")
    working = population.copy()
    working["timestamp_utc"] = pd.to_datetime(
        working["timestamp_utc"], utc=True, errors="raise"
    )
    working["admissible_at_utc"] = pd.to_datetime(
        working["admissible_at_utc"], utc=True, errors="raise"
    )
    folds: list[Fold] = []
    first_day = working["timestamp_utc"].min().ceil("D")
    last_day = working["timestamp_utc"].max().floor("D")
    starts = pd.date_range(
        first_day, last_day, freq=f"{design.test_step_hours}h", tz="UTC"
    )
    for start in starts:
        end = start + pd.Timedelta(hours=design.test_block_hours)
        train = working.loc[working["admissible_at_utc"] <= start]
        test = working.loc[
            working["timestamp_utc"].ge(start) & working["timestamp_utc"].lt(end)
        ]
        if train.empty or test.empty:
            continue
        span_hours = (start - train["timestamp_utc"].min()).total_seconds() / 3600
        positives = int(train["label"].sum())
        negatives = len(train) - positives
        if (
            len(train) < design.minimum_training_rows
            or span_hours < design.minimum_training_hours
            or positives < design.minimum_training_positives
            or negatives < design.minimum_training_negatives
        ):
            continue
        folds.append(
            Fold(
                index=len(folds),
                test_start_utc=start,
                test_end_exclusive_utc=end,
                training_index=tuple(int(value) for value in train.index),
                test_index=tuple(int(value) for value in test.index),
            )
        )
    if not folds:
        raise ValidationError("no estimable daily fold exists")
    return tuple(folds)


def validate_temporal_separation(
    population: pd.DataFrame, folds: tuple[Fold, ...]
) -> None:
    """Verify that every training outcome precedes its test block."""

    for fold in folds:
        train = population.loc[list(fold.training_index)]
        test = population.loc[list(fold.test_index)]
        if train["admissible_at_utc"].max() > fold.test_start_utc:
            raise ValidationError("a fold admits information after test start")
        if test["timestamp_utc"].min() < fold.test_start_utc:
            raise ValidationError("a test row precedes its fold")
        if test["timestamp_utc"].max() >= fold.test_end_exclusive_utc:
            raise ValidationError("a test row exceeds its fold")
