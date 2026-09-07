"""Fold-causal calibration based on earlier out-of-sample predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stablecoin_ews.contracts import STUDY_ASSETS, StudyDesign


@dataclass(frozen=True)
class CalibrationSummary:
    """Readiness and history counts for one prediction block."""

    ready: bool
    reasons: tuple[str, ...]
    history_rows: int
    history_positives: int
    history_negatives: int
    history_calendar_days: float
    rows_by_asset: dict[str, int]


def _design_matrix(frame: pd.DataFrame, margin_column: str) -> np.ndarray:
    return np.column_stack(
        [
            frame[margin_column].to_numpy(dtype=float),
            *[
                frame["asset_id"].eq(asset).to_numpy(dtype=float)
                for asset in STUDY_ASSETS
            ],
        ]
    )


def _fit_platt(
    history: pd.DataFrame, current: pd.DataFrame, margin_column: str
) -> np.ndarray:
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        fit_intercept=False,
        max_iter=2_000,
        tol=1e-8,
        random_state=42,
    )
    model.fit(_design_matrix(history, margin_column), history["label"].astype(int))
    if model.coef_.shape != (1, 4) or model.coef_[0, 0] <= 0:
        raise ValueError("Platt calibration must have a positive common slope")
    return model.predict_proba(_design_matrix(current, margin_column))[:, 1]


def calibrate_prediction_block(
    earlier_oos: pd.DataFrame | None,
    current_oos: pd.DataFrame,
    test_start_utc: object,
    design: StudyDesign | None = None,
) -> tuple[pd.DataFrame, CalibrationSummary]:
    """Calibrate a block using only sufficiently mature earlier OOS outcomes."""

    design = design or StudyDesign()
    current = current_oos.copy()
    target_start = pd.Timestamp(test_start_utc)
    if target_start.tzinfo is None:
        target_start = target_start.tz_localize("UTC")
    else:
        target_start = target_start.tz_convert("UTC")
    if earlier_oos is None or earlier_oos.empty:
        history = current.iloc[:0].copy()
    else:
        history = earlier_oos.copy()
        label_end = pd.to_datetime(
            history["label_observation_end_utc"], utc=True, errors="raise"
        )
        mature = label_end + pd.Timedelta(hours=design.calibration_embargo_hours)
        history = history.loc[mature <= target_start].copy()
    history = history.sort_values(["timestamp_utc", "asset_id"], kind="mergesort")
    labels = history["label"].to_numpy(dtype=int) if len(history) else np.array([])
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    counts = {asset: int(history["asset_id"].eq(asset).sum()) for asset in STUDY_ASSETS}
    span_days = 0.0
    if len(history):
        timestamps = pd.to_datetime(history["timestamp_utc"], utc=True)
        span_days = (timestamps.max() - timestamps.min()).total_seconds() / 86_400
    reasons: list[str] = []
    if negatives < design.minimum_calibration_negatives:
        reasons.append("insufficient_negatives")
    if positives < design.minimum_calibration_positives:
        reasons.append("insufficient_positives")
    if span_days < design.minimum_calibration_days:
        reasons.append("insufficient_calendar_span")
    if any(
        count < design.minimum_calibration_rows_per_asset for count in counts.values()
    ):
        reasons.append("insufficient_rows_for_receiving_asset")
    current["calibration_ready"] = not reasons
    current["calibration_reason"] = "ready" if not reasons else "+".join(reasons)
    current["historical_prevalence_probability"] = np.nan
    for asset in STUDY_ASSETS:
        asset_history = history.loc[history["asset_id"].eq(asset), "label"]
        if not asset_history.empty:
            current.loc[
                current["asset_id"].eq(asset), "historical_prevalence_probability"
            ] = float(asset_history.mean())
    for estimator in ("logistic", "lightgbm"):
        output_column = f"{estimator}_calibrated_probability"
        current[output_column] = np.nan
        if not reasons:
            current[output_column] = _fit_platt(
                history, current, f"{estimator}_raw_margin"
            )
    summary = CalibrationSummary(
        ready=not reasons,
        reasons=tuple(reasons),
        history_rows=len(history),
        history_positives=positives,
        history_negatives=negatives,
        history_calendar_days=float(span_days),
        rows_by_asset=counts,
    )
    return current, summary
