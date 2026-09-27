"""Unit tests for P3 pair features."""

import unittest

from src.features import (
    NormalizedRecord,
    address_similarity,
    build_pair_features,
    iter_feature_rows,
    name_similarity,
    normalize_record,
    token_jaccard,
)


def record(name=None, address=None, country=None):
    return {
        "business_name": name,
        "business_address": address,
        "country": country,
    }


class SimilarityHelperTests(unittest.TestCase):
    def test_token_jaccard(self):
        self.assertAlmostEqual(token_jaccard("acme services", "acme group"), 1 / 3)
        self.assertEqual(token_jaccard("", "acme"), 0.0)

    def test_name_similarity_identical_similar_and_different(self):
        self.assertEqual(name_similarity("acme llc", "acme llc"), 1.0)
        self.assertGreater(name_similarity("acme services", "acme service"), 0.0)
        self.assertLess(name_similarity("acme services", "unrelated clinic"), 1.0)

    def test_address_similarity_is_bounded(self):
        self.assertEqual(address_similarity("1 main road", "1 main road"), 1.0)
        self.assertGreaterEqual(address_similarity("1 main road", "2 oak street"), 0.0)
        self.assertLessEqual(address_similarity("1 main road", "2 oak street"), 1.0)


class PairFeatureTests(unittest.TestCase):
    def test_identical_name_and_address(self):
        features = build_pair_features(
            record("Acme Inc", "1 Main Road", "US"),
            record("ACME INC", "1 Main Road", "us"),
            "S2",
        )
        self.assertTrue(features["name_exact"])
        self.assertTrue(features["address_exact"])
        self.assertTrue(features["country_exact"])
        self.assertEqual(features["name_similarity"], 1.0)
        self.assertEqual(features["address_similarity"], 1.0)

    def test_similar_names_and_different_names(self):
        similar = build_pair_features(record("Cobalt LLC"), record("Cobalt Ltd"), "S2")
        different = build_pair_features(record("Cobalt LLC"), record("Unrelated Clinic"), "S2")
        self.assertFalse(similar["name_exact"])
        self.assertGreater(similar["name_similarity"], 0.0)
        self.assertLess(
            float(different["name_similarity"]),
            float(similar["name_similarity"]),
        )

    def test_missing_addresses_are_unknown_not_a_mismatch(self):
        features = build_pair_features(record("Acme", None), record("Acme", "  "), "S2")
        self.assertTrue(features["address_missing"])
        self.assertFalse(features["address_exact"])
        self.assertEqual(features["address_similarity"], 0.0)

    def test_different_addresses_are_not_exact(self):
        features = build_pair_features(
            record("Acme", "1 Main Road"),
            record("Acme", "99 Oak Avenue"),
            "S2",
        )
        self.assertFalse(features["address_exact"])
        self.assertLess(features["address_similarity"], 1.0)

    def test_missing_names_and_values_are_safe(self):
        features = build_pair_features(record(None, None), record(" ", " "), "S3")
        self.assertTrue(features["name_missing"])
        self.assertFalse(features["name_exact"])
        self.assertEqual(features["name_similarity"], 0.0)

    def test_country_mismatch_is_explicit(self):
        features = build_pair_features(record("Acme", country="US"), record("Acme", country="France"), "S3")
        self.assertFalse(features["country_exact"])
        self.assertTrue(features["country_mismatch"])

    def test_all_similarity_values_are_bounded(self):
        features = build_pair_features(
            record("Acme Group", "1 Main Road", "US"),
            record("Acme Holdings", "2 Oak Street", "India"),
            "S2",
        )
        for key, value in features.items():
            if "similarity" in key or "jaccard" in key:
                self.assertGreaterEqual(float(value), 0.0)
                self.assertLessEqual(float(value), 1.0)

    def test_cached_record_and_pair_iterator(self):
        normalized = normalize_record(record("Acme LLC", "1 Main Road", "US"))
        self.assertIsInstance(normalized, NormalizedRecord)
        records = {
            "S1-1": normalized,
            "S2-1": normalize_record(record("Acme LLC", "1 Main Road", "US")),
        }
        pairs = [{
            "s1_entity_id": "S1-1",
            "candidate_entity_id": "S2-1",
            "candidate_source": "S2",
            "blocking_method": "exact_normalized_name",
        }]
        row = next(iter_feature_rows(pairs, records))
        self.assertEqual(row["s1_entity_id"], "S1-1")
        self.assertTrue(row["name_exact"])
        self.assertEqual(row["blocking_method"], "exact_normalized_name")


if __name__ == "__main__":
    unittest.main()
