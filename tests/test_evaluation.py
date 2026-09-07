import unittest

import numpy as np
import pandas as pd

from stablecoin_ews.evaluation import (
    alert_grid,
    associate_alerts_to_events,
    persistent_alert_hours,
    predictive_metrics,
    threshold_metrics,
)
from stablecoin_ews.uncertainty import (
    paired_moving_block_bootstrap,
    percentile_interval,
)


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_threshold_grid(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.7, 0.9])
        baseline = np.full(4, 0.5)
        metrics = predictive_metrics(labels, probabilities, baseline)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(len(threshold_metrics(labels, probabilities)), 8)

    def test_persistence_resets_on_time_gap(self) -> None:
        frame = pd.DataFrame(
            {
                "asset_id": ["dai"] * 4,
                "timestamp_utc": pd.to_datetime(
                    [
                        "2024-01-01T00:00:00Z",
                        "2024-01-01T01:00:00Z",
                        "2024-01-01T03:00:00Z",
                        "2024-01-01T04:00:00Z",
                    ]
                ),
                "probability": [0.3, 0.4, 0.5, 0.6],
                "label": [0, 0, 1, 1],
            }
        )
        alerts = persistent_alert_hours(frame, "probability", 0.2, 2)
        self.assertEqual(len(alerts), 2)
        self.assertEqual(len(alert_grid(frame, "probability")), 24)

    def test_event_association_uses_pre_onset_window(self) -> None:
        eligible = pd.DataFrame(
            {
                "asset_id": "dai",
                "timestamp_utc": pd.date_range(
                    "2024-01-01T00:00:00Z", periods=25, freq="h"
                ),
            }
        )
        alerts = pd.DataFrame(
            {
                "asset_id": ["dai"],
                "alert_hour_utc": [pd.Timestamp("2024-01-01T10:00:00Z")],
            }
        )
        events = pd.DataFrame(
            {
                "asset_id": ["dai"],
                "onset_utc": [pd.Timestamp("2024-01-02T00:00:00Z")],
            }
        )
        association = associate_alerts_to_events(alerts, events, eligible)
        self.assertEqual(association.loc[0, "coverage"], "full")
        self.assertTrue(association.loc[0, "detected"])
        self.assertEqual(association.loc[0, "lead_time_hours"], 14)

    def test_paired_bootstrap_is_reproducible(self) -> None:
        frame = pd.DataFrame(
            {
                "asset_id": ["dai"] * 8,
                "timestamp_utc": pd.date_range(
                    "2024-01-01", periods=8, freq="h", tz="UTC"
                ),
                "label": [0, 0, 0, 1, 0, 1, 0, 1],
                "m0": np.linspace(0.1, 0.8, 8),
                "m1": np.linspace(0.05, 0.9, 8),
            }
        )

        def brier(labels: np.ndarray, probability: np.ndarray) -> float:
            return float(np.mean((labels - probability) ** 2))

        first = paired_moving_block_bootstrap(
            frame, "m0", "m1", brier, replicates=20, block_hours=3
        )
        second = paired_moving_block_bootstrap(
            frame, "m0", "m1", brier, replicates=20, block_hours=3
        )
        pd.testing.assert_frame_equal(first, second)
        lower, upper = percentile_interval(first["m1_minus_m0"])
        self.assertLessEqual(lower, upper)


if __name__ == "__main__":
    unittest.main()
