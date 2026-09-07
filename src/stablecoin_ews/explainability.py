"""Fold-local TreeSHAP calculation and additivity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ExplainabilityError(ValueError):
    """Raised when fold-local attribution output is inconsistent."""


def fold_local_tree_shap(
    fitted_lightgbm: object,
    test_matrix: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Explain the fitted fold model in raw-margin space without refitting it."""

    try:
        import shap
    except ImportError as exc:
        raise ExplainabilityError(
            "SHAP is required for fold-local explanations"
        ) from exc
    matrix = np.asarray(test_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ExplainabilityError("test matrix and feature schema differ")
    booster = getattr(fitted_lightgbm, "booster_", fitted_lightgbm)
    explainer = shap.TreeExplainer(
        booster,
        data=None,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    values = explainer.shap_values(matrix, check_additivity=False)
    if isinstance(values, list):
        values = values[-1]
    shap_matrix = np.asarray(values, dtype=float)
    if shap_matrix.ndim == 3 and shap_matrix.shape[-1] == 2:
        shap_matrix = shap_matrix[:, :, 1]
    if shap_matrix.shape != matrix.shape:
        raise ExplainabilityError("SHAP output shape differs from the test matrix")
    expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    base_value = float(expected[-1])
    margin = np.asarray(booster.predict(matrix, raw_score=True), dtype=float)
    residual = base_value + shap_matrix.sum(axis=1) - margin
    if not np.isfinite(shap_matrix).all() or not np.isfinite(residual).all():
        raise ExplainabilityError("SHAP output contains a non-finite value")
    if (np.abs(residual) > tolerance).mean() > 0.0001:
        raise ExplainabilityError("SHAP additivity failed for too many rows")
    output = pd.DataFrame(
        shap_matrix, columns=[f"shap__{name}" for name in feature_names]
    )
    output.insert(0, "additivity_pass", np.abs(residual) <= tolerance)
    output.insert(0, "additivity_residual", residual)
    output.insert(0, "base_raw_margin", base_value)
    return output
