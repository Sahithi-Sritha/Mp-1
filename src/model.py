"""Explainable deterministic scoring for P2 candidate pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from src.features import (
    CandidatePair,
    FeatureRow,
    RecordLookup,
    iter_feature_rows,
)


@dataclass(frozen=True)
class ScoringConfig:
    """Editable evidence weights, exact-rule cutoffs, and decision threshold."""

    name_exact_weight: float = 0.25
    name_core_exact_weight: float = 0.20
    name_similarity_weight: float = 0.16
    name_core_similarity_weight: float = 0.08
    name_token_jaccard_weight: float = 0.10
    address_exact_weight: float = 0.10
    address_similarity_weight: float = 0.06
    address_token_jaccard_weight: float = 0.03
    country_exact_weight: float = 0.02
    country_mismatch_penalty: float = 0.12
    conflicting_exact_address_penalty: float = 0.18
    very_high_name_similarity: float = 0.96
    very_high_address_similarity: float = 0.96
    exact_name_country_score: float = 0.99
    exact_core_address_score: float = 0.97
    very_high_pair_score: float = 0.94
    threshold: float = 0.75


DEFAULT_CONFIG = ScoringConfig()


def score_features(
    features: Mapping[str, str | float | bool | int],
    config: ScoringConfig = DEFAULT_CONFIG,
) -> float:
    """Return a confidence score in ``0..1`` using rules then weighted evidence."""
    name_exact = bool(features.get("name_exact", False))
    core_exact = bool(features.get("name_core_exact", False))
    address_exact = bool(features.get("address_exact", False))
    country_exact = bool(features.get("country_exact", False))
    country_mismatch = bool(features.get("country_mismatch", False))
    name_similarity = float(features.get("name_similarity", 0.0))
    address_similarity = float(features.get("address_similarity", 0.0))

    # High-confidence rules run before the softer weighted score.
    if name_exact and country_exact:
        score = config.exact_name_country_score
    elif core_exact and address_exact:
        score = config.exact_core_address_score
    elif (
        name_similarity >= config.very_high_name_similarity
        and address_similarity >= config.very_high_address_similarity
    ):
        score = config.very_high_pair_score
    else:
        score = (
            config.name_exact_weight * name_exact
            + config.name_core_exact_weight * core_exact
            + config.name_similarity_weight * name_similarity
            + config.name_core_similarity_weight
            * float(features.get("name_core_similarity", 0.0))
            + config.name_token_jaccard_weight
            * float(features.get("name_token_jaccard", 0.0))
            + config.address_exact_weight * address_exact
            + config.address_similarity_weight * address_similarity
            + config.address_token_jaccard_weight
            * float(features.get("address_token_jaccard", 0.0))
            + config.country_exact_weight * country_exact
        )

    # A country discrepancy reduces confidence but never rejects by itself.
    if country_mismatch:
        score -= config.country_mismatch_penalty

    # Penalize a concrete contradiction: exactly matching address but strongly
    # conflicting names. Unknown/missing address is not negative evidence.
    if address_exact and name_similarity < 0.15 and not (name_exact or core_exact):
        score -= config.conflicting_exact_address_penalty

    return min(1.0, max(0.0, score))


def predict_match(
    features: Mapping[str, str | float | bool | int],
    threshold: float | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> dict[str, str | float | bool]:
    """Return the score and independent threshold decision for one pair."""
    cutoff = config.threshold if threshold is None else threshold
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    score = score_features(features, config)
    result: dict[str, str | float | bool] = {
        "match_score": score,
        "predicted_match": score >= cutoff,
    }
    for key in (
        "s1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "blocking_method",
    ):
        if key in features:
            result[key] = str(features[key])
    return result


def score_candidate(
    features: Mapping[str, str | float | bool | int],
    config: ScoringConfig = DEFAULT_CONFIG,
) -> dict[str, str | float | bool]:
    """Compatibility wrapper returning the requested IDs and model fields."""
    return predict_match(features, config=config)


def iter_scored_candidates(
    candidate_pairs: Iterable[CandidatePair],
    records_by_id: RecordLookup,
    config: ScoringConfig = DEFAULT_CONFIG,
    threshold: float | None = None,
    cache_size: int = 25_000,
) -> Iterable[dict[str, str | float | bool]]:
    """Yield a result for every candidate without choosing a single winner.

    Pairs are handled independently and in input order. No Cartesian product is
    constructed; feature generation uses a bounded normalization cache.
    """
    features = iter_feature_rows(candidate_pairs, records_by_id, cache_size)
    for feature_row in features:
        yield predict_match(feature_row, threshold=threshold, config=config)
