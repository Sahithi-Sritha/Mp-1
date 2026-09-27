"""Candidate generation for business entity resolution."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from src.normalize import (
    normalize_business_name,
    normalize_business_name_core,
)


REQUIRED_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)


def build_name_index(
    records: Iterable[Mapping[str, str]],
) -> dict[str, list[str]]:
    """Build an index from normalized business name to entity IDs.

    Empty normalized names are ignored so missing names do not create
    candidates between unrelated records.
    """
    index: dict[str, list[str]] = defaultdict(list)

    for record in records:
        entity_id = record["entity_id"]
        business_name = record["business_name"]

        normalized_name = normalize_business_name(business_name)

        if not normalized_name:
            continue

        index[normalized_name].append(entity_id)

    return dict(index)


def build_core_name_index(
    records: Iterable[Mapping[str, str]],
) -> dict[str, list[str]]:
    """Build an index using normalized business-name cores.

    Recognized legal suffixes at the end of a business name are removed.
    Empty core names are ignored.
    """
    index: dict[str, list[str]] = defaultdict(list)

    for record in records:
        entity_id = record["entity_id"]
        business_name = record["business_name"]

        normalized_core = normalize_business_name_core(business_name)

        if not normalized_core:
            continue

        index[normalized_core].append(entity_id)

    return dict(index)


def generate_name_candidates(
    source1_records: Iterable[Mapping[str, str]],
    candidate_records: Iterable[Mapping[str, str]],
    candidate_source: str,
) -> list[dict[str, str]]:
    """Generate candidates using exact normalized business-name matching.

    Parameters
    ----------
    source1_records:
        Records from S1.

    candidate_records:
        Records from either S2 or S3.

    candidate_source:
        Source label, for example ``"S2"`` or ``"S3"``.

    Returns
    -------
    list[dict[str, str]]
        Candidate pairs containing the S1 ID, candidate ID,
        candidate source, and blocking method.
    """
    index = build_name_index(candidate_records)

    candidates: set[tuple[str, str, str, str]] = set()

    for record in source1_records:
        s1_id = record["entity_id"]
        normalized_name = normalize_business_name(
            record["business_name"]
        )

        if not normalized_name:
            continue

        for candidate_id in index.get(normalized_name, []):
            candidates.add(
                (
                    s1_id,
                    candidate_id,
                    candidate_source,
                    "exact_normalized_name",
                )
            )

    return [
        {
            "s1_entity_id": s1_id,
            "candidate_entity_id": candidate_id,
            "candidate_source": source,
            "blocking_method": method,
        }
        for s1_id, candidate_id, source, method in sorted(candidates)
    ]


def generate_core_name_candidates(
    source1_records: Iterable[Mapping[str, str]],
    candidate_records: Iterable[Mapping[str, str]],
    candidate_source: str,
) -> list[dict[str, str]]:
    """Generate candidates using exact normalized business-name cores.

    Legal suffixes such as ``Inc``, ``Ltd``, ``LLC``, and ``Corp`` are
    removed when they occur at the end of the business name.
    """
    index = build_core_name_index(candidate_records)

    candidates: set[tuple[str, str, str, str]] = set()

    for record in source1_records:
        s1_id = record["entity_id"]
        normalized_core = normalize_business_name_core(
            record["business_name"]
        )

        if not normalized_core:
            continue

        for candidate_id in index.get(normalized_core, []):
            candidates.add(
                (
                    s1_id,
                    candidate_id,
                    candidate_source,
                    "exact_normalized_name_core",
                )
            )

    return [
        {
            "s1_entity_id": s1_id,
            "candidate_entity_id": candidate_id,
            "candidate_source": source,
            "blocking_method": method,
        }
        for s1_id, candidate_id, source, method in sorted(candidates)
    ]