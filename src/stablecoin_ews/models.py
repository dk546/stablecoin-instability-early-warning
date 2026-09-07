"""Prespecified logistic and gradient-boosted fold models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from stablecoin_ews.contracts import SourceFamily, feature_columns
from stablecoin_ews.validation import Fold


class ModelError(ValueError):
    """Raised when a fold cannot support the prespecified model pair."""


@dataclass
class FittedFold:
    """Fitted objects and out-of-sample predictions for one fold."""

    fold: Fold
    source_family: SourceFamily
    feature_names: tuple[str, ...]
    scaler: StandardScaler
    logistic: LogisticRegression
    lightgbm: Any
    predictions: pd.DataFrame
    test_matrix: np.ndarray


def training_weights(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Create equal-asset weights and balanced final effective weights."""

    if frame.empty:
        raise ModelError("training population is empty")
    assets = frame["asset_id"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=int)
    unique_assets, asset_counts = np.unique(assets, return_counts=True)
    classes, class_counts = np.unique(labels, return_counts=True)
    if tuple(classes.tolist()) != (0, 1):
        raise ModelError("training data must contain both outcome classes")
    asset_count = dict(zip(unique_assets, asset_counts, strict=True))
    class_count = dict(zip(classes, class_counts, strict=True))
    equal_asset = np.asarray(
        [len(frame) / (len(unique_assets) * asset_count[asset]) for asset in assets],
        dtype=float,
    )
    class_factor = {value: len(frame) / (2.0 * class_count[value]) for value in (0, 1)}
    raw = equal_asset * np.asarray([class_factor[value] for value in labels])
    scale = len(frame) / raw.sum()
    logistic_sample_weight = equal_asset * scale
    final_effective_weight = logistic_sample_weight * np.asarray(
        [class_factor[value] for value in labels]
    )
    if not np.isclose(final_effective_weight.mean(), 1.0, atol=1e-12, rtol=0):
        raise ModelError("training weights do not have unit mean")
    return logistic_sample_weight, final_effective_weight


def fit_fold_models(
    population: pd.DataFrame,
    fold: Fold,
    source_family: SourceFamily | str,
) -> FittedFold:
    """Fit both estimators on one fold and return native OOS outputs."""

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ModelError("LightGBM is required to fit the model pair") from exc
    family = SourceFamily(source_family)
    columns = feature_columns(family)
    missing = set(columns).difference(population.columns)
    if missing:
        raise ModelError(f"model frame is missing features: {sorted(missing)}")
    train = population.loc[list(fold.training_index)].sort_values(
        ["admissible_at_utc", "asset_id", "timestamp_utc"], kind="mergesort"
    )
    test = population.loc[list(fold.test_index)].sort_values(
        ["asset_id", "timestamp_utc"], kind="mergesort"
    )
    x_train = train.loc[:, columns].to_numpy(dtype=float, copy=True)
    x_test = test.loc[:, columns].to_numpy(dtype=float, copy=True)
    y_train = train["label"].to_numpy(dtype=int, copy=True)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ModelError("model matrices must be finite")
    logistic_weight, tree_weight = training_weights(train)
    class_counts = np.bincount(y_train, minlength=2)
    class_factors = {
        value: len(y_train) / (2.0 * class_counts[value]) for value in (0, 1)
    }
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    logistic = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2_000,
        tol=1e-8,
        class_weight=class_factors,
        random_state=42,
    )
    logistic.fit(x_train_scaled, y_train, sample_weight=logistic_weight)
    lightgbm = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=42,
        deterministic=True,
        force_col_wise=True,
        n_jobs=4,
        verbosity=-1,
    )
    lightgbm.fit(x_train, y_train, sample_weight=tree_weight)
    logistic_margin = logistic.decision_function(x_test_scaled).astype(float)
    logistic_probability = logistic.predict_proba(x_test_scaled)[:, 1].astype(float)
    tree_margin = lightgbm.booster_.predict(x_test, raw_score=True).astype(float)
    tree_probability = lightgbm.booster_.predict(x_test, raw_score=False).astype(float)
    if not all(
        np.isfinite(values).all()
        for values in (
            logistic_margin,
            logistic_probability,
            tree_margin,
            tree_probability,
        )
    ):
        raise ModelError("model output contains a non-finite value")
    predictions = test.loc[
        :, ["asset_id", "timestamp_utc", "label", "label_observation_end_utc"]
    ].copy()
    predictions["fold_index"] = fold.index
    predictions["source_family"] = family.value
    predictions["logistic_raw_margin"] = logistic_margin
    predictions["logistic_native_probability"] = logistic_probability
    predictions["lightgbm_raw_margin"] = tree_margin
    predictions["lightgbm_native_probability"] = tree_probability
    return FittedFold(
        fold=fold,
        source_family=family,
        feature_names=columns,
        scaler=scaler,
        logistic=logistic,
        lightgbm=lightgbm,
        predictions=predictions.reset_index(drop=True),
        test_matrix=x_test,
    )
