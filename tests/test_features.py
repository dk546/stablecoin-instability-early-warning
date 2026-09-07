import unittest

from stablecoin_ews.contracts import M0_FEATURES, M1_FEATURES
from stablecoin_ews.features import build_feature_frames
from stablecoin_ews.synthetic import synthetic_hourly_inputs


class FeatureTests(unittest.TestCase):
    def test_synthetic_features_match_both_schemas(self) -> None:
        market, macro = synthetic_hourly_inputs(hours=360)
        m0, m1 = build_feature_frames(market, macro)
        self.assertEqual(len(m0), 1_080)
        self.assertEqual(len(m1), 1_080)
        self.assertTrue(set(M0_FEATURES).issubset(m0.columns))
        self.assertTrue(set(M1_FEATURES).issubset(m1.columns))
        self.assertGreater(int(m0["feature_eligibility"].sum()), 0)
        self.assertGreater(int(m1["feature_eligibility"].sum()), 0)
        self.assertLessEqual(
            int(m1["feature_eligibility"].sum()),
            int(m0["feature_eligibility"].sum()),
        )


if __name__ == "__main__":
    unittest.main()
