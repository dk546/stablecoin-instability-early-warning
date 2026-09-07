import unittest

import numpy as np
import pandas as pd

from stablecoin_ews.calibration import calibrate_prediction_block
from stablecoin_ews.contracts import M0_FEATURES, STUDY_ASSETS, StudyDesign
from stablecoin_ews.validation import (
    expanding_daily_folds,
    model_population,
    validate_temporal_separation,
)


class ValidationAndCalibrationTests(unittest.TestCase):
    def test_expanding_folds_respect_outcome_availability(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
        labels = pd.DataFrame(
            {
                "asset_id": "dai",
                "timestamp_utc": timestamps,
                "label": [index % 29 == 0 for index in range(96)],
                "prediction_outcome_eligible": True,
                "label_observation_end_utc": timestamps + pd.Timedelta(hours=25),
            }
        )
        features = labels.loc[:, ["asset_id", "timestamp_utc"]].copy()
        for feature in M0_FEATURES:
            features[feature] = 1.0
        features["feature_eligibility"] = True
        population = model_population(
            labels,
            features,
            StudyDesign(training_embargo_hours=1),
        )
        design = StudyDesign(
            training_embargo_hours=1,
            minimum_training_hours=24,
            minimum_training_rows=24,
        )
        folds = expanding_daily_folds(population, design)
        validate_temporal_separation(population, folds)
        self.assertGreater(len(folds), 0)

    @staticmethod
    def prediction_history() -> pd.DataFrame:
        rows = []
        for asset_index, asset in enumerate(STUDY_ASSETS):
            for day in range(4):
                label = int((asset_index + day) % 3 == 0)
                rows.append(
                    {
                        "asset_id": asset,
                        "timestamp_utc": pd.Timestamp("2024-01-01T00:00:00Z")
                        + pd.Timedelta(days=day),
                        "label_observation_end_utc": pd.Timestamp(
                            "2024-01-01T01:00:00Z"
                        )
                        + pd.Timedelta(days=day),
                        "label": label,
                        "logistic_raw_margin": -1.5 + 3.0 * label + day / 100,
                        "lightgbm_raw_margin": -1.2 + 2.4 * label + day / 100,
                    }
                )
        return pd.DataFrame(rows)

    def test_causal_calibration_and_historical_baseline(self) -> None:
        history = self.prediction_history()
        current = history.iloc[:3].copy()
        design = StudyDesign(
            calibration_embargo_hours=0,
            minimum_calibration_negatives=1,
            minimum_calibration_positives=1,
            minimum_calibration_days=0,
            minimum_calibration_rows_per_asset=1,
        )
        output, summary = calibrate_prediction_block(
            history,
            current,
            pd.Timestamp("2024-02-01T00:00:00Z"),
            design,
        )
        self.assertTrue(summary.ready)
        self.assertTrue(np.isfinite(output["logistic_calibrated_probability"]).all())
        self.assertTrue(np.isfinite(output["historical_prevalence_probability"]).all())

    def test_unready_history_is_not_mislabeled_as_calibrated(self) -> None:
        history = self.prediction_history().iloc[:2]
        current = self.prediction_history().iloc[:3].copy()
        output, summary = calibrate_prediction_block(
            history,
            current,
            pd.Timestamp("2024-02-01T00:00:00Z"),
        )
        self.assertFalse(summary.ready)
        self.assertTrue(output["logistic_calibrated_probability"].isna().all())


if __name__ == "__main__":
    unittest.main()
