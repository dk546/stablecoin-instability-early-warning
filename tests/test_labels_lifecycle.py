import unittest

import pandas as pd

from stablecoin_ews.labels import build_warning_labels, detect_event_episodes
from stablecoin_ews.lifecycle import observed_support_intervals


class LabelAndLifecycleTests(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        timestamps = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
        prices = [1.0] * 40
        prices[12:15] = [0.988, 0.987, 0.986]
        return pd.DataFrame(
            {
                "asset_id": "dai",
                "timestamp_utc": timestamps,
                "price_observed": True,
                "price_usd": prices,
            }
        )

    def test_persistent_breach_forms_one_episode(self) -> None:
        episodes = detect_event_episodes(self.frame())
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].observed_hours, 3)
        self.assertEqual(episodes[0].onset_utc, pd.Timestamp("2024-01-01T12:00:00Z"))

    def test_warning_label_and_active_event_exclusion(self) -> None:
        labels, _ = build_warning_labels(self.frame())
        first = labels.loc[
            labels["timestamp_utc"].eq(pd.Timestamp("2024-01-01T00:00:00Z"))
        ].iloc[0]
        active = labels.loc[
            labels["timestamp_utc"].eq(pd.Timestamp("2024-01-01T12:00:00Z"))
        ].iloc[0]
        self.assertEqual(first["label"], 1)
        self.assertEqual(first["label_status"], "positive")
        self.assertEqual(active["label_status"], "active_event_excluded")
        self.assertTrue(pd.isna(active["label"]))

    def test_observed_support_is_split_by_gap(self) -> None:
        frame = self.frame()
        frame.loc[5, "price_observed"] = False
        intervals = observed_support_intervals(frame)
        self.assertEqual([value.observed_hours for value in intervals], [5, 34])


if __name__ == "__main__":
    unittest.main()
