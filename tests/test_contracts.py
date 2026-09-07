import math
import unittest

from stablecoin_ews.contracts import (
    M0_FEATURES,
    M1_FEATURES,
    ContractError,
    SourceFamily,
    assess_feature_record,
    feature_columns,
    validate_feature_order,
)


class ContractTests(unittest.TestCase):
    def test_feature_families_have_fixed_nested_order(self) -> None:
        self.assertEqual(len(M0_FEATURES), 22)
        self.assertEqual(len(M1_FEATURES), 26)
        self.assertEqual(M1_FEATURES[:22], M0_FEATURES)
        self.assertEqual(feature_columns(SourceFamily.M0_MARKET), M0_FEATURES)

    def test_feature_eligibility_requires_finite_values(self) -> None:
        record = {name: 1.0 for name in M1_FEATURES}
        eligible, invalid = assess_feature_record(record, SourceFamily.M1_MARKET_MACRO)
        self.assertTrue(eligible)
        self.assertEqual(invalid, ())
        record[M0_FEATURES[3]] = math.nan
        eligible, invalid = assess_feature_record(record, SourceFamily.M0_MARKET)
        self.assertFalse(eligible)
        self.assertEqual(invalid, (M0_FEATURES[3],))

    def test_reordered_matrix_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            validate_feature_order(tuple(reversed(M0_FEATURES)), SourceFamily.M0_MARKET)


if __name__ == "__main__":
    unittest.main()
