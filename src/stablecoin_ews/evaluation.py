"""Predictive metrics and prespecified operational alert evaluation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from stablecoin_ews.contracts import (
    PERSISTENCE_HOURS,
    PROBABILITY_THRESHOLDS,
    StudyDesign,
)


def _calibration_diagnostic(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) != 2:
        return None, None
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2_000,
        tol=1e-8,
        random_state=42,
    ).fit(logits, labels)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def predictive_metrics(
    labels: object, probabilities: object, baseline_probabilities: object
) -> dict[str, float | int | None]:
    """Calculate discrimination and calibration measures for one population."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    baseline = np.asarray(baseline_probabilities, dtype=float)
    if not len(y) or not np.isfinite(p).all() or not np.isfinite(baseline).all():
        raise ValueError("metric inputs must be non-empty and finite")
    positives = int(y.sum())
    negatives = len(y) - positives
    prevalence = float(y.mean())
    average_precision = float(average_precision_score(y, p)) if positives else None
    roc_auc = float(roc_auc_score(y, p)) if positives and negatives else None
    brier = float(brier_score_loss(y, p))
    baseline_brier = float(brier_score_loss(y, baseline))
    intercept, slope = _calibration_diagnostic(y, p)
    return {
        "row_count": len(y),
        "positive_count": positives,
        "negative_count": negatives,
        "prevalence": prevalence,
        "average_precision": average_precision,
        "average_precision_lift": (
            average_precision / prevalence
            if average_precision is not None and prevalence > 0
            else None
        ),
        "roc_auc": roc_auc,
        "brier_score": brier,
        "historical_prevalence_brier_score": baseline_brier,
        "historical_prevalence_brier_skill": (
            None if baseline_brier == 0 else 1 - brier / baseline_brier
        ),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def threshold_metrics(
    labels: object,
    probabilities: object,
    thresholds: tuple[float, ...] = PROBABILITY_THRESHOLDS,
) -> pd.DataFrame:
    """Report precision, recall, and F1 over the prespecified threshold grid."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    rows: list[dict[str, float | int | None]] = []
    for threshold in thresholds:
        predicted = p >= threshold
        true_positive = int((predicted & (y == 1)).sum())
        false_positive = int((predicted & (y == 0)).sum())
        false_negative = int(((~predicted) & (y == 1)).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        rows.append(
            {
                "threshold": threshold,
                "true_positives": true_positive,
                "false_positives": false_positive,
                "false_negatives": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def persistent_alert_hours(
    predictions: pd.DataFrame,
    probability_column: str,
    threshold: float,
    persistence_hours: int,
) -> pd.DataFrame:
    """Emit hours after a within-asset threshold run reaches persistence."""

    required = {"asset_id", "timestamp_utc", probability_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction data are missing columns: {sorted(missing)}")
    if persistence_hours < 1:
        raise ValueError("persistence must be positive")
    frame = predictions.copy()
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="raise"
    )
    emitted: list[dict[str, object]] = []
    for asset, group in frame.groupby("asset_id", sort=True):
        group = group.sort_values("timestamp_utc", kind="mergesort")
        run = 0
        previous: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            timestamp = row.timestamp_utc
            probability = float(getattr(row, probability_column))
            contiguous = previous is not None and timestamp - previous == pd.Timedelta(
                hours=1
            )
            run = (
                run + 1
                if contiguous and probability >= threshold
                else int(probability >= threshold)
            )
            if run >= persistence_hours:
                emitted.append(
                    {
                        "asset_id": str(asset),
                        "alert_hour_utc": timestamp,
                        "threshold": threshold,
                        "persistence_hours": persistence_hours,
                    }
                )
            previous = timestamp
    return pd.DataFrame(
        emitted,
        columns=[
            "asset_id",
            "alert_hour_utc",
            "threshold",
            "persistence_hours",
        ],
    )


def alert_grid(
    predictions: pd.DataFrame,
    probability_column: str,
    design: StudyDesign | None = None,
) -> pd.DataFrame:
    """Summarize warning burden and row performance over the complete grid."""

    design = design or StudyDesign()
    required = {"asset_id", "timestamp_utc", "label", probability_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction data are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for threshold in PROBABILITY_THRESHOLDS:
        for persistence in PERSISTENCE_HOURS:
            alerts = persistent_alert_hours(
                predictions, probability_column, threshold, persistence
            )
            alert_keys = set(
                zip(
                    alerts.get("asset_id", []),
                    alerts.get("alert_hour_utc", []),
                    strict=True,
                )
            )
            emitted = np.asarray(
                [
                    (str(row.asset_id), pd.Timestamp(row.timestamp_utc)) in alert_keys
                    for row in predictions.itertuples(index=False)
                ],
                dtype=bool,
            )
            labels = predictions["label"].to_numpy(dtype=int)
            true_positive = int((emitted & (labels == 1)).sum())
            false_positive = int((emitted & (labels == 0)).sum())
            false_negative = int(((~emitted) & (labels == 1)).sum())
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else None
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else None
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None
                and recall is not None
                and precision + recall > 0
                else None
            )
            rows.append(
                {
                    "threshold": threshold,
                    "persistence_hours": persistence,
                    "reference": (
                        math.isclose(threshold, design.reference_probability_threshold)
                        and persistence == design.reference_persistence_hours
                    ),
                    "eligible_hours": len(predictions),
                    "alert_hours": int(emitted.sum()),
                    "true_positive_alert_hours": true_positive,
                    "false_alert_hours": false_positive,
                    "false_negative_hours": false_negative,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_alert_hours_per_eligible_asset_month": (
                        false_positive * 730.485 / len(predictions)
                    ),
                    "time_in_warning_fraction": float(emitted.mean()),
                }
            )
    return pd.DataFrame(rows)


def associate_alerts_to_events(
    alerts: pd.DataFrame,
    events: pd.DataFrame,
    eligible_predictions: pd.DataFrame,
    design: StudyDesign | None = None,
) -> pd.DataFrame:
    """Associate alert hours with each event's 24-to-6-hour pre-onset window."""

    design = design or StudyDesign()
    alert_required = {"asset_id", "alert_hour_utc"}
    event_required = {"asset_id", "onset_utc"}
    eligible_required = {"asset_id", "timestamp_utc"}
    if missing := alert_required.difference(alerts.columns):
        raise ValueError(f"alert data are missing columns: {sorted(missing)}")
    if missing := event_required.difference(events.columns):
        raise ValueError(f"event data are missing columns: {sorted(missing)}")
    if missing := eligible_required.difference(eligible_predictions.columns):
        raise ValueError(f"eligible data are missing columns: {sorted(missing)}")
    alert_frame = alerts.copy()
    event_frame = events.copy()
    eligible = eligible_predictions.copy()
    alert_frame["alert_hour_utc"] = pd.to_datetime(
        alert_frame["alert_hour_utc"], utc=True, errors="raise"
    )
    event_frame["onset_utc"] = pd.to_datetime(
        event_frame["onset_utc"], utc=True, errors="raise"
    )
    eligible["timestamp_utc"] = pd.to_datetime(
        eligible["timestamp_utc"], utc=True, errors="raise"
    )
    rows: list[dict[str, object]] = []
    for event_index, event in enumerate(event_frame.itertuples(index=False)):
        onset = event.onset_utc
        start = onset - pd.Timedelta(hours=design.warning_end_hours)
        end = onset - pd.Timedelta(hours=design.warning_start_hours)
        expected = set(pd.date_range(start, end, freq="h", tz="UTC"))
        observed = set(
            eligible.loc[
                eligible["asset_id"].eq(event.asset_id)
                & eligible["timestamp_utc"].between(start, end, inclusive="both"),
                "timestamp_utc",
            ]
        )
        qualifying = sorted(
            set(
                alert_frame.loc[
                    alert_frame["asset_id"].eq(event.asset_id)
                    & alert_frame["alert_hour_utc"].between(
                        start, end, inclusive="both"
                    ),
                    "alert_hour_utc",
                ]
            )
        )
        coverage = "full" if observed == expected else "partial" if observed else "none"
        rows.append(
            {
                "event_index": event_index,
                "asset_id": str(event.asset_id),
                "onset_utc": onset,
                "coverage": coverage,
                "eligible_hour_count": len(observed),
                "expected_hour_count": len(expected),
                "detected": bool(qualifying),
                "earliest_alert_utc": qualifying[0] if qualifying else pd.NaT,
                "lead_time_hours": (
                    (onset - qualifying[0]).total_seconds() / 3600
                    if qualifying
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def equal_frequency_reliability(
    labels: object, probabilities: object, bins: int = 10
) -> pd.DataFrame:
    """Create a deterministic equal-frequency reliability summary."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or bins < 2:
        raise ValueError("reliability inputs are inconsistent")
    order = np.lexsort((np.arange(len(p)), p))
    groups = np.array_split(order, bins)
    return pd.DataFrame(
        [
            {
                "bin": index,
                "row_count": len(group),
                "mean_probability": float(p[group].mean()) if len(group) else None,
                "observed_rate": float(y[group].mean()) if len(group) else None,
            }
            for index, group in enumerate(groups)
        ]
    )
