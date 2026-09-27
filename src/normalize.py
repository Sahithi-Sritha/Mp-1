"""Conservative text normalization helpers for business entity resolution."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal


MissingValueKind = Literal["blank", "placeholder", "value"]

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_SEPARATORS = frozenset(".,;:!?&'’‘\"“”()[]{}\\|~`…")


def _nfkc_casefold(text: str) -> str:
    """Apply compatibility Unicode normalization and case folding."""
    return unicodedata.normalize("NFKC", text).casefold()


def normalize_text(text: str | None) -> str:
    """Normalize text while retaining numbers, scripts, and token boundaries.

    Most punctuation becomes a space so punctuation differences do not merge
    adjacent words. Hyphens, slashes, and hash marks are retained because they
    can be meaningful in names, street numbers, and unit identifiers.
    """
    if text is None:
        return ""

    normalized = _nfkc_casefold(text)
    output: list[str] = []
    for char in normalized:
        # Normalize all Unicode punctuation consistently, but retain common
        # address/name separators that can carry meaning.
        category = unicodedata.category(char)
        if char in _PUNCTUATION_SEPARATORS or category.startswith("P"):
            if char not in "#/-":
                output.append(" ")
                continue
        output.append(char)

    normalized = re.sub(r"([#/-])\1+", r"\1", "".join(output))
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_business_name(text: str | None) -> str:
    """Normalize a business name, keeping its legal suffix tokens."""
    if text is None:
        return ""
    # An ampersand is a common spelling of “and” in business names.
    return normalize_text(text.replace("&", " and ").replace("＆", " and "))


_LEGAL_SUFFIXES: tuple[tuple[str, ...], ...] = (
    ("private", "limited"),
    ("private", "ltd"),
    ("pvt", "limited"),
    ("pvt", "ltd"),
    ("l", "l", "c"),
    ("l", "l", "p"),
    ("limited",),
    ("ltd",),
    ("private",),
    ("pvt",),
    ("llc",),
    ("inc",),
    ("incorporated",),
    ("corporation",),
    ("corp",),
    ("llp",),
    ("company",),
    ("co",),
    ("sarl",),
    ("sas",),
)


def normalize_business_name_core(text: str | None) -> str:
    """Return a comparison view with recognized legal suffixes removed.

    Only exact suffix tokens at the end are removed. The base normalized name
    remains available through :func:`normalize_business_name`.
    """
    tokens = normalize_business_name(text).split()
    while tokens:
        suffix = next(
            (
                candidate
                for candidate in _LEGAL_SUFFIXES
                if len(tokens) > len(candidate)
                and tuple(tokens[-len(candidate) :]) == candidate
            ),
            None,
        )
        if suffix is None:
            break
        del tokens[-len(suffix) :]
    return " ".join(tokens)


def normalize_address(text: str | None) -> str:
    """Normalize address typography without dropping address components."""
    return normalize_text(text)


def normalize_country(text: str | None) -> str:
    """Normalize country label casing and whitespace without a country list."""
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", _nfkc_casefold(text)).strip()


def classify_missing_value(text: str | None) -> MissingValueKind:
    """Classify blank, literal ``null`` placeholder, or ordinary text.

    The literal word ``null`` is reported as ``placeholder``; it is not
    silently converted to an empty string by the normalization functions.
    Other strings, including ``NA`` and ``unknown``, remain ordinary values.
    """
    if text is None or not text.strip():
        return "blank"
    if _nfkc_casefold(text).strip() == "null":
        return "placeholder"
    return "value"


if __name__ == "__main__":
    examples = [
        (
            "Kelly Advisory, Inc",
            "1795 Westchester Drive, High Point, NC",
        ),
        (
            "Callicoat & Dailey Inc",
            "2505, Tower 1, Oakwood, Runwal Greens, Mumbai, Maharashtra",
        ),
        (
            "Healthcare Janki Nutrition Pvt. Ltd.",
            "H.No.16-11-23/37/A, 2nd Floor, Flat No.207, Hyderabad, Telangana",
        ),
        (
            "Quartz L.L.C.",
            "914 Pierpont Ave, Unit 11, Cleveland, OH",
        ),
        (
            "Saint-Herblain Societe SARL",
            "175 Boulevard du Président Franklin Roosevelt, Bordeaux, France",
        ),
        (
            "M/s Sandeep Software (India) Pvt. Ltd",
            "G-3/571, Gulmohar Colony, Bhopal, Madhya Pradesh",
        ),
    ]

    print("raw name | normalized name | core name | raw address | normalized address")
    for name, address in examples:
        print(
            f"{name} | {normalize_business_name(name)} | "
            f"{normalize_business_name_core(name)} | {address} | "
            f"{normalize_address(address)}"
        )
