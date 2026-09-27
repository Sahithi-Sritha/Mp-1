"""Lightweight features for candidate pairs emitted by :mod:`src.blocking`."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Mapping, TypedDict

from src.normalize import (
    normalize_address,
    normalize_business_name,
    normalize_business_name_core,
    normalize_country,
)


class CandidatePair(TypedDict, total=False):
    """Candidate dictionary returned by the P2 generators."""

    s1_entity_id: str
    candidate_entity_id: str
    candidate_source: str
    blocking_method: str


Record = Mapping[str, str | None]
FeatureRow = dict[str, str | float | bool | int]


@dataclass(frozen=True)
class NormalizedRecord:
    """Normalized values cached once for reuse across candidate pairs."""

    business_name: str
    business_name_core: str
    business_address: str
    country: str


RecordLookup = Mapping[str, Record | NormalizedRecord]


def normalize_record(record: Record) -> NormalizedRecord:
    """Normalize the four P2 record fields once for subsequent comparisons."""
    name = _value(record, "business_name")
    return NormalizedRecord(
        business_name=normalize_business_name(name),
        business_name_core=normalize_business_name_core(name),
        business_address=normalize_address(_value(record, "business_address")),
        country=normalize_country(_value(record, "country")),
    )


def _value(record: Record, column: str) -> str:
    value = record.get(column)
    return value if isinstance(value, str) else ""


def token_jaccard(left: str, right: str) -> float:
    """Return token-set Jaccard similarity; blank inputs produce zero."""
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _character_ngram_jaccard(left: str, right: str, n: int = 3) -> float:
    """Compare short character n-grams using a bounded, linear-time set score."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    # For very short values, individual characters give useful overlap.
    n = min(n, len(left), len(right))
    left_grams = {left[i : i + n] for i in range(len(left) - n + 1)}
    right_grams = {right[i : i + n] for i in range(len(right) - n + 1)}
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 0.0


def _similarity(left: str, right: str) -> float:
    """Blend token and character n-gram overlap into a score in ``0..1``."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    score = 0.65 * token_jaccard(left, right) + 0.35 * _character_ngram_jaccard(
        left, right
    )
    return min(1.0, max(0.0, score))


def name_similarity(left: str, right: str) -> float:
    """Return fast token/character overlap for normalized business names."""
    return _similarity(left, right)


def address_similarity(left: str, right: str) -> float:
    """Return fast token/character overlap for normalized addresses."""
    return _similarity(left, right)


def build_pair_features(
    s1_record: Record | NormalizedRecord,
    candidate_record: Record | NormalizedRecord,
    candidate_source: str,
    s1_entity_id: str | None = None,
    candidate_entity_id: str | None = None,
    blocking_method: str | None = None,
) -> FeatureRow:
    """Build the requested features for one pair of raw or cached records."""
    s1 = s1_record if isinstance(s1_record, NormalizedRecord) else normalize_record(s1_record)
    candidate = (
        candidate_record
        if isinstance(candidate_record, NormalizedRecord)
        else normalize_record(candidate_record)
    )

    name_missing = not s1.business_name or not candidate.business_name
    address_missing = not s1.business_address or not candidate.business_address
    name_jaccard = token_jaccard(s1.business_name, candidate.business_name)
    address_jaccard = token_jaccard(s1.business_address, candidate.business_address)

    features: FeatureRow = {
        "candidate_source": candidate_source,
        "name_exact": bool(
            not name_missing and s1.business_name == candidate.business_name
        ),
        "name_similarity": name_similarity(
            s1.business_name, candidate.business_name
        ),
        "name_core_exact": bool(
            s1.business_name_core
            and candidate.business_name_core
            and s1.business_name_core == candidate.business_name_core
        ),
        "name_core_similarity": name_similarity(
            s1.business_name_core, candidate.business_name_core
        ),
        "name_token_jaccard": name_jaccard,
        "name_length_diff": abs(len(s1.business_name) - len(candidate.business_name)),
        "address_exact": bool(
            not address_missing and s1.business_address == candidate.business_address
        ),
        "address_similarity": address_similarity(
            s1.business_address, candidate.business_address
        ),
        "address_token_jaccard": address_jaccard,
        "address_length_diff": abs(
            len(s1.business_address) - len(candidate.business_address)
        ),
        "address_missing": address_missing,
        "country_exact": bool(
            s1.country and candidate.country and s1.country == candidate.country
        ),
        "country_mismatch": bool(
            s1.country and candidate.country and s1.country != candidate.country
        ),
        "name_missing": name_missing,
    }
    if s1_entity_id is not None:
        features["s1_entity_id"] = s1_entity_id
    if candidate_entity_id is not None:
        features["candidate_entity_id"] = candidate_entity_id
    if blocking_method is not None:
        features["blocking_method"] = blocking_method
    return features


# Backwards-compatible function name from the initial P3 draft.
calculate_pair_features = build_pair_features


def iter_feature_rows(
    candidate_pairs: Iterable[CandidatePair],
    records_by_id: RecordLookup,
    cache_size: int = 25_000,
) -> Iterable[FeatureRow]:
    """Yield features for P2 pairs, normalizing each recently used record once.

    The bounded LRU cache avoids repeated normalization for common S1 and
    candidate records without growing with the full dataset. A caller that
    already has cached :class:`NormalizedRecord` values may supply them in the
    same ID-to-record mapping. The input pair stream is never materialized.
    """
    if cache_size < 0:
        raise ValueError("cache_size must be non-negative")
    cache: OrderedDict[str, NormalizedRecord] = OrderedDict()

    def cached(entity_id: str) -> NormalizedRecord:
        if entity_id in cache:
            cache.move_to_end(entity_id)
            return cache[entity_id]
        record = records_by_id[entity_id]
        normalized = (
            record if isinstance(record, NormalizedRecord) else normalize_record(record)
        )
        if cache_size:
            cache[entity_id] = normalized
            if len(cache) > cache_size:
                cache.popitem(last=False)
        return normalized

    for pair in candidate_pairs:
        s1_id = pair["s1_entity_id"]
        candidate_id = pair["candidate_entity_id"]
        yield build_pair_features(
            cached(s1_id),
            cached(candidate_id),
            candidate_source=pair["candidate_source"],
            s1_entity_id=s1_id,
            candidate_entity_id=candidate_id,
            blocking_method=pair.get("blocking_method"),
        )
