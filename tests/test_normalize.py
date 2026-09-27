"""Focused tests for the reusable P1 normalization helpers."""

import unittest

from src.normalize import (
    classify_missing_value,
    normalize_address,
    normalize_business_name,
    normalize_business_name_core,
    normalize_country,
    normalize_text,
)


class NormalizeTextTests(unittest.TestCase):
    def test_casefolds_uppercase_and_lowercase(self) -> None:
        self.assertEqual(normalize_text("MiXeD Case"), "mixed case")

    def test_collapses_extra_whitespace(self) -> None:
        self.assertEqual(normalize_text("  Acme\t\n  Services  "), "acme services")

    def test_punctuation_becomes_token_boundaries(self) -> None:
        self.assertEqual(normalize_text("Kelly Advisory, Inc."), "kelly advisory inc")

    def test_nfkc_normalizes_compatibility_characters(self) -> None:
        self.assertEqual(normalize_text("ＡＣＭＥ　１２３"), "acme 123")

    def test_empty_and_none_are_safe(self) -> None:
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(" \t\n "), "")
        self.assertEqual(normalize_text(None), "")

    def test_keeps_non_latin_text_and_numbers(self) -> None:
        self.assertEqual(normalize_text("दिल्ली 560001"), "दिल्ली 560001")


class BusinessNameTests(unittest.TestCase):
    def test_kelly_name_punctuation_and_case(self) -> None:
        self.assertEqual(normalize_business_name("Kelly Advisory, Inc"), "kelly advisory inc")

    def test_ampersand_is_normalized_to_and(self) -> None:
        self.assertEqual(
            normalize_business_name("Callicoat & Dailey Inc"),
            "callicoat and dailey inc",
        )

    def test_legal_suffix_remains_in_normal_representation(self) -> None:
        self.assertEqual(
            normalize_business_name("Healthcare Janki Nutrition Pvt. Ltd."),
            "healthcare janki nutrition pvt ltd",
        )

    def test_spacing_variations_collapse(self) -> None:
        self.assertEqual(
            normalize_business_name("  North   Star\t Holdings  LLC "),
            "north star holdings llc",
        )


class BusinessNameCoreTests(unittest.TestCase):
    def test_removes_recognized_trailing_suffixes(self) -> None:
        cases = {
            "Kelly Advisory, Inc": "kelly advisory",
            "Acme Private Limited": "acme",
            "Quartz L.L.C.": "quartz",
            "Example Group SARL": "example group",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_business_name_core(raw), expected)

    def test_does_not_remove_the_entire_name(self) -> None:
        self.assertEqual(normalize_business_name_core("LLC"), "llc")
        self.assertEqual(normalize_business_name_core("Limited"), "limited")

    def test_does_not_remove_arbitrary_trailing_words(self) -> None:
        self.assertEqual(
            normalize_business_name_core("Acme Services Group"),
            "acme services group",
        )

    def test_suffix_is_removed_only_at_the_end(self) -> None:
        self.assertEqual(
            normalize_business_name_core("Inc Services Acme"),
            "inc services acme",
        )


class AddressTests(unittest.TestCase):
    def test_preserves_numbers_and_number_separators(self) -> None:
        self.assertEqual(
            normalize_address("H.No.16-11-23/37/A, 2nd Floor"),
            "h no 16-11-23/37/a 2nd floor",
        )

    def test_preserves_apartment_and_unit_information(self) -> None:
        normalized = normalize_address("Tower 1, Unit 11, Flat No.207, Apt 4B")
        self.assertEqual(normalized, "tower 1 unit 11 flat no 207 apt 4b")

    def test_normalizes_punctuation_and_whitespace(self) -> None:
        self.assertEqual(
            normalize_address("  1795 Westchester Drive,,,  High Point, NC  "),
            "1795 westchester drive high point nc",
        )

    def test_keeps_meaningful_address_tokens(self) -> None:
        self.assertEqual(
            normalize_address("Near Fortis Hospital #803, Sector 5"),
            "near fortis hospital #803 sector 5",
        )

    def test_empty_and_none_are_safe(self) -> None:
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(normalize_address(None), "")


class CountryTests(unittest.TestCase):
    def test_normalizes_case_and_whitespace(self) -> None:
        self.assertEqual(normalize_country("  InDiA\t "), "india")

    def test_accepts_countries_without_a_hardcoded_allowlist(self) -> None:
        self.assertEqual(normalize_country("  Côte   d’Ivoire "), "côte d’ivoire")
        self.assertEqual(normalize_country("日本"), "日本")

    def test_none_is_safe(self) -> None:
        self.assertEqual(normalize_country(None), "")


class MissingValueTests(unittest.TestCase):
    def test_empty_string_is_blank(self) -> None:
        self.assertEqual(classify_missing_value(""), "blank")

    def test_whitespace_only_string_is_blank(self) -> None:
        self.assertEqual(classify_missing_value(" \t\n "), "blank")

    def test_literal_null_is_a_placeholder(self) -> None:
        self.assertEqual(classify_missing_value("null"), "placeholder")
        self.assertEqual(classify_missing_value(" NULL "), "placeholder")

    def test_normal_non_empty_value_is_value(self) -> None:
        self.assertEqual(classify_missing_value("France"), "value")

    def test_none_is_blank(self) -> None:
        self.assertEqual(classify_missing_value(None), "blank")


if __name__ == "__main__":
    unittest.main()
