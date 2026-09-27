"""Unit tests for the deterministic P3 matching scorer."""

import unittest

from src.features import build_pair_features
from src.model import ScoringConfig, predict_match, score_features


def pair_features(name1, address1, country1, name2, address2, country2):
    return build_pair_features(
        {"business_name": name1, "business_address": address1, "country": country1},
        {"business_name": name2, "business_address": address2, "country": country2},
        "S2",
        "S1-1",
        "S2-1",
    )


class ModelTests(unittest.TestCase):
    def test_strong_exact_match_is_predicted(self):
        result = predict_match(
            pair_features("Acme Inc", "1 Main Road", "US", "Acme Inc", "1 Main Road", "US")
        )
        self.assertTrue(result["predicted_match"])
        self.assertGreaterEqual(result["match_score"], 0.75)

    def test_clearly_different_pair_is_not_predicted(self):
        features = pair_features(
            "Cobalt LLC", "1 Main Road", "US", "Unrelated Clinic", "99 Oak Ave", "US"
        )
        result = predict_match(features)
        self.assertFalse(result["predicted_match"])

    def test_missing_address_does_not_automatically_reject_exact_name(self):
        features = pair_features("Acme Inc", None, "US", "Acme Inc", "", "US")
        self.assertTrue(features["address_missing"])
        self.assertTrue(predict_match(features)["predicted_match"])

    def test_country_mismatch_lowers_confidence(self):
        same_country = pair_features("Acme Inc", "1 Main Road", "US", "Acme Inc", "1 Main Road", "US")
        mismatch = pair_features("Acme Inc", "1 Main Road", "US", "Acme Inc", "1 Main Road", "India")
        self.assertLess(score_features(mismatch), score_features(same_country))

    def test_multiple_candidates_are_scored_independently(self):
        candidates = [
            pair_features("Acme Inc", "1 Main Road", "US", "Acme Inc", "1 Main Road", "US"),
            pair_features("Acme Inc", "1 Main Road", "US", "Different Co", "99 Oak Ave", "US"),
        ]
        results = [predict_match(row) for row in candidates]
        self.assertEqual(len(results), 2)
        self.assertEqual([r["predicted_match"] for r in results], [True, False])

    def test_threshold_is_configurable(self):
        features = pair_features("Acme Inc", "1 Main Road", "US", "Acme Inc", "1 Main Road", "US")
        self.assertTrue(predict_match(features, threshold=0.90)["predicted_match"])
        self.assertFalse(predict_match(features, threshold=1.0)["predicted_match"])
        self.assertTrue(predict_match(features, config=ScoringConfig(threshold=0.90))["predicted_match"])

    def test_score_is_clipped_to_zero_to_one(self):
        weak = {"name_similarity": 0.0, "country_mismatch": True}
        strong = {
            "name_exact": True,
            "name_core_exact": True,
            "name_similarity": 1.0,
            "name_core_similarity": 1.0,
            "name_token_jaccard": 1.0,
            "address_exact": True,
            "address_similarity": 1.0,
            "address_token_jaccard": 1.0,
            "country_exact": True,
        }
        self.assertGreaterEqual(score_features(weak), 0.0)
        self.assertLessEqual(score_features(strong), 1.0)


if __name__ == "__main__":
    unittest.main()
