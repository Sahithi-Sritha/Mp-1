"""Tests for business-name candidate blocking."""

import unittest

from src.blocking import (
    build_core_name_index,
    build_name_index,
    generate_core_name_candidates,
    generate_name_candidates,
)


def make_record(
    entity_id: str,
    business_name: str,
    business_address: str = "",
    country: str = "",
) -> dict[str, str]:
    """Create a record using the project source schema."""
    return {
        "entity_id": entity_id,
        "business_name": business_name,
        "business_address": business_address,
        "country": country,
    }


class BuildNameIndexTests(unittest.TestCase):
    def test_normalizes_business_names_before_indexing(self) -> None:
        records = [
            make_record("S2-1", "Kelly Advisory, Inc."),
            make_record("S2-2", "Prime Money"),
        ]

        index = build_name_index(records)

        self.assertEqual(index["kelly advisory inc"], ["S2-1"])
        self.assertEqual(index["prime money"], ["S2-2"])

    def test_preserves_multiple_entities_with_same_name(self) -> None:
        records = [
            make_record("S2-1", "Acme Services"),
            make_record("S2-2", "ACME SERVICES"),
        ]

        index = build_name_index(records)

        self.assertEqual(index["acme services"], ["S2-1", "S2-2"])

    def test_ignores_empty_business_names(self) -> None:
        records = [
            make_record("S2-1", ""),
            make_record("S2-2", "   "),
            make_record("S2-3", "Valid Business"),
        ]

        index = build_name_index(records)

        self.assertNotIn("", index)
        self.assertEqual(index["valid business"], ["S2-3"])


class GenerateNameCandidatesTests(unittest.TestCase):
    def test_generates_exact_normalized_name_candidates(self) -> None:
        source1 = [
            make_record("S1-1", "Kelly Advisory, Inc."),
            make_record("S1-2", "Unrelated Business"),
        ]

        source2 = [
            make_record("S2-1", "kelly advisory inc"),
            make_record("S2-2", "Another Business"),
        ]

        candidates = generate_name_candidates(
            source1,
            source2,
            "S2",
        )

        self.assertEqual(
            candidates,
            [
                {
                    "s1_entity_id": "S1-1",
                    "candidate_entity_id": "S2-1",
                    "candidate_source": "S2",
                    "blocking_method": "exact_normalized_name",
                }
            ],
        )

    def test_generates_multiple_candidates_for_one_s1(self) -> None:
        source1 = [
            make_record("S1-1", "Acme Services"),
        ]

        source2 = [
            make_record("S2-1", "Acme Services"),
            make_record("S2-2", "ACME SERVICES"),
            make_record("S2-3", "Different Business"),
        ]

        candidates = generate_name_candidates(
            source1,
            source2,
            "S2",
        )

        candidate_ids = {
            candidate["candidate_entity_id"]
            for candidate in candidates
        }

        self.assertEqual(candidate_ids, {"S2-1", "S2-2"})

    def test_does_not_generate_candidates_for_empty_s1_name(self) -> None:
        source1 = [
            make_record("S1-1", ""),
        ]

        source2 = [
            make_record("S2-1", ""),
            make_record("S2-2", "Valid Business"),
        ]

        candidates = generate_name_candidates(
            source1,
            source2,
            "S2",
        )

        self.assertEqual(candidates, [])

    def test_deduplicates_candidate_pairs(self) -> None:
        source1 = [
            make_record("S1-1", "Acme Services"),
        ]

        source2 = [
            make_record("S2-1", "Acme Services"),
        ]

        candidates = generate_name_candidates(
            source1,
            source2,
            "S2",
        )

        self.assertEqual(len(candidates), 1)

    def test_supports_s3_as_candidate_source(self) -> None:
        source1 = [
            make_record("S1-1", "Prime Money"),
        ]

        source3 = [
            make_record("S3-1", "PRIME MONEY"),
        ]

        candidates = generate_name_candidates(
            source1,
            source3,
            "S3",
        )

        self.assertEqual(
            candidates,
            [
                {
                    "s1_entity_id": "S1-1",
                    "candidate_entity_id": "S3-1",
                    "candidate_source": "S3",
                    "blocking_method": "exact_normalized_name",
                }
            ],
        )


class CoreNameBlockingTests(unittest.TestCase):
    def test_removes_legal_suffix_for_indexing(self) -> None:
        records = [
            make_record("S2-1", "Kelly Advisory, Inc."),
            make_record("S2-2", "Prime Money Ltd."),
        ]

        index = build_core_name_index(records)

        self.assertEqual(index["kelly advisory"], ["S2-1"])
        self.assertEqual(index["prime money"], ["S2-2"])

    def test_matches_different_legal_suffixes(self) -> None:
        source1 = [
            make_record("S1-1", "Acme Services Inc."),
        ]

        source2 = [
            make_record("S2-1", "Acme Services Ltd."),
        ]

        candidates = generate_core_name_candidates(
            source1,
            source2,
            "S2",
        )

        self.assertEqual(
            candidates,
            [
                {
                    "s1_entity_id": "S1-1",
                    "candidate_entity_id": "S2-1",
                    "candidate_source": "S2",
                    "blocking_method": "exact_normalized_name_core",
                }
            ],
        )

    def test_does_not_use_empty_core_name(self) -> None:
        source1 = [
            make_record("S1-1", ""),
        ]

        source2 = [
            make_record("S2-1", ""),
        ]

        candidates = generate_core_name_candidates(
            source1,
            source2,
            "S2",
        )

        self.assertEqual(candidates, [])

    def test_core_name_handles_punctuation_and_case(self) -> None:
        source1 = [
            make_record("S1-1", "Quartz L.L.C."),
        ]

        source2 = [
            make_record("S2-1", "quartz llc"),
        ]

        candidates = generate_core_name_candidates(
            source1,
            source2,
            "S2",
        )

        self.assertEqual(
            candidates,
            [
                {
                    "s1_entity_id": "S1-1",
                    "candidate_entity_id": "S2-1",
                    "candidate_source": "S2",
                    "blocking_method": "exact_normalized_name_core",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()