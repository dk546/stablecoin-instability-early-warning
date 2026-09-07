import unittest

import numpy as np
import pandas as pd

from stablecoin_ews.contracts import M0_FEATURES, SourceFamily
from stablecoin_ews.explainability import fold_local_tree_shap
from stablecoin_ews.models import fit_fold_models, training_weights
from stablecoin_ews.validation import Fold


class ModelAndExplainabilityTests(unittest.TestCase):
    @staticmethod
    def population() -> pd.DataFrame:
        rng = np.random.default_rng(42)
        row_count = 120
        frame = pd.DataFrame(
            {
                "asset_id": np.resize(
                    np.array(["dai", "tether", "usd-coin"]), row_count
                ),
                "timestamp_utc": pd.date_range(
                    "2023-01-01", periods=row_count, freq="h", tz="UTC"
                ),
                "label_observation_end_utc": pd.date_range(
                    "2023-01-02", periods=row_count, freq="h", tz="UTC"
                ),
                "admissible_at_utc": pd.date_range(
                    "2023-01-03", periods=row_count, freq="h", tz="UTC"
                ),
            }
        )
        for index, feature in enumerate(M0_FEATURES):
            frame[feature] = rng.normal(index / 10, 1, row_count)
        frame["label"] = (frame[M0_FEATURES[0]] + frame[M0_FEATURES[1]] > 0).astype(int)
        return frame

    def test_weighting_and_fold_local_shap(self) -> None:
        population = self.population()
        logistic_weight, tree_weight = training_weights(population.iloc[:90])
        self.assertTrue(np.isfinite(logistic_weight).all())
        self.assertAlmostEqual(float(tree_weight.mean()), 1.0)
        fold = Fold(
            index=0,
            test_start_utc=pd.Timestamp("2023-01-06T00:00:00Z"),
            test_end_exclusive_utc=pd.Timestamp("2023-01-07T00:00:00Z"),
            training_index=tuple(range(90)),
            test_index=tuple(range(90, 120)),
        )
        fitted = fit_fold_models(population, fold, SourceFamily.M0_MARKET)
        self.assertEqual(len(fitted.predictions), 30)
        explanations = fold_local_tree_shap(
            fitted.lightgbm, fitted.test_matrix, fitted.feature_names
        )
        self.assertEqual(len(explanations), 30)
        self.assertTrue(explanations["additivity_pass"].all())


if __name__ == "__main__":
    unittest.main()
