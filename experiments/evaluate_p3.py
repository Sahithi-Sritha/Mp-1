"""Evaluate P3 on sampled S1 records and a bounded training candidate pool.

Example: ``py -m experiments.evaluate_p3 --sample-size 25``

Each S2/S3 file is read once. The evaluator retains a bounded prefix as the
negative candidate pool and additionally retains only rows whose IDs are true
matches for sampled S1 records. Other rows are not normalized or kept. Blocking
recall is measured against sampled truths; P3 classification metrics use only
true matches that survived bounded P2 candidate generation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from src.blocking import generate_core_name_candidates, generate_name_candidates
from src.features import build_pair_features, normalize_record
from src.model import DEFAULT_CONFIG, score_features


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "student_resource" / "dataset" / "train"
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
DEFAULT_CANDIDATE_POOL_SIZE = 50_000
SOURCE_FILES = {
    "S2": TRAIN / "train_source2.tsv",
    "S3": TRAIN / "train_source3.tsv",
}


def read_sample(path: Path, limit: int) -> list[dict[str, str]]:
    """Read at most ``limit`` TSV records from the start of a file."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return [dict(row) for _, row in zip(range(limit), reader)]


def load_truth_for_sample(
    path: Path, sampled_s1_ids: set[str]
) -> dict[str, set[str]]:
    """Stream ground truth and retain labels only for sampled Source 1 IDs."""
    truth = {entity_id: set() for entity_id in sampled_s1_ids}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            if s1_id not in truth:
                continue
            raw_ids = row.get("matched_entity_ids", "") or ""
            truth[s1_id].update(
                matched_id.strip()
                for matched_id in raw_ids.split(",")
                if matched_id.strip()
            )
    return truth


def load_bounded_candidate_pool(
    path: Path, limit: int, required_ids: set[str]
) -> tuple[dict[str, dict[str, str]], int]:
    """Read a source once, retaining the prefix and exact needed target rows.

    The returned dictionary is keyed by entity ID. Rows already in the prefix
    satisfy a required target ID without being inserted a second time. Reading
    stops only when the prefix is full and every required ID has been found.
    """
    records_by_id: dict[str, dict[str, str]] = {}
    found_required: set[str] = set()
    prefix_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            entity_id = row["entity_id"]

            # Always take the bounded prefix for negative candidate examples.
            if prefix_count < limit:
                records_by_id[entity_id] = dict(row)
                prefix_count += 1

            # Beyond that prefix, retain only IDs needed to label true pairs.
            if entity_id in required_ids:
                found_required.add(entity_id)
                records_by_id.setdefault(entity_id, dict(row))

            if prefix_count >= limit and found_required == required_ids:
                break

    missing = required_ids - found_required
    if missing:
        example_ids = ", ".join(sorted(missing)[:5])
        raise RuntimeError(
            f"Could not find {len(missing)} required true-match IDs in {path.name}; "
            f"examples: {example_ids}"
        )

    added_target_count = len(records_by_id) - prefix_count
    return records_by_id, added_target_count


def f05(tp: int, fp: int, fn: int) -> float:
    """Compute F-beta with beta=0.5 from contingency counts."""
    denominator = 1.25 * tp + 0.25 * fn + fp
    return 1.25 * tp / denominator if denominator else 0.0


def evaluate(
    sample_size: int,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    print(f"Reading {sample_size} sampled S1 records...", flush=True)
    s1_records = read_sample(TRAIN / "train_source1.tsv", sample_size)
    if not s1_records:
        raise RuntimeError("No S1 records were read; check the training data path.")
    s1_ids = {row["entity_id"] for row in s1_records}
    print(f"Sampled S1 records: {len(s1_records)}", flush=True)

    print("Loading ground truth for sampled S1 IDs...", flush=True)
    truth = load_truth_for_sample(TRAIN / "train_ground_truth.tsv", s1_ids)
    true_pairs = {
        (s1_id, candidate_id)
        for s1_id, candidate_ids in truth.items()
        for candidate_id in candidate_ids
    }
    if len(true_pairs) == 0:
        print("Loaded ground truth: zero true matches in this S1 sample.", flush=True)
    else:
        print(f"Loaded ground truth: {len(true_pairs)} true matches.", flush=True)

    true_ids_by_source = {source: set() for source in SOURCE_FILES}
    for _, candidate_id in true_pairs:
        if candidate_id.startswith("S2-"):
            true_ids_by_source["S2"].add(candidate_id)
        elif candidate_id.startswith("S3-"):
            true_ids_by_source["S3"].add(candidate_id)
        else:
            raise RuntimeError(f"Unexpected ground-truth candidate ID: {candidate_id}")

    print(
        f"Loading bounded candidate prefixes ({candidate_pool_size} rows/source) "
        "and retrieving only sampled true-match IDs...",
        flush=True,
    )
    source_records: dict[str, dict[str, dict[str, str]]] = {}
    for source, path in SOURCE_FILES.items():
        print(f"Reading bounded candidate pool from {path.name}...", flush=True)
        source_records[source], added_targets = load_bounded_candidate_pool(
            path, candidate_pool_size, true_ids_by_source[source]
        )
        loaded_true = len(true_ids_by_source[source])
        print(
            f"Loaded {len(source_records[source])} candidate records and "
            f"{loaded_true} required true-match records "
            f"({added_targets} added beyond the prefix).",
            flush=True,
        )

    # Exact name/core P2 blocking is applied only to the bounded candidate pool.
    candidate_pairs: dict[tuple[str, str], dict[str, str]] = {}
    for source, candidates in source_records.items():
        for generator in (generate_name_candidates, generate_core_name_candidates):
            for pair in generator(s1_records, candidates.values(), source):
                candidate_pairs.setdefault(
                    (pair["s1_entity_id"], pair["candidate_entity_id"]), pair
                )
    candidate_keys = set(candidate_pairs)
    true_candidates = candidate_keys & true_pairs
    print(f"Generated bounded P2 candidates: {len(candidate_pairs)}", flush=True)

    records_by_id = {
        row["entity_id"]: normalize_record(row)
        for row in (
            *s1_records,
            *source_records["S2"].values(),
            *source_records["S3"].values(),
        )
    }
    print("Generating features and scores once for each candidate...", flush=True)
    scored: list[tuple[float, bool]] = []
    for pair in candidate_pairs.values():
        feature_row = build_pair_features(
            records_by_id[pair["s1_entity_id"]],
            records_by_id[pair["candidate_entity_id"]],
            pair["candidate_source"],
            pair["s1_entity_id"],
            pair["candidate_entity_id"],
            pair.get("blocking_method"),
        )
        is_true_candidate = (
            pair["s1_entity_id"], pair["candidate_entity_id"]
        ) in true_candidates
        scored.append((score_features(feature_row), is_true_candidate))

    print(f"Generated features/scores: {len(scored)}", flush=True)
    print("Evaluating thresholds...", flush=True)
    table: list[dict[str, float | int]] = []
    for threshold in THRESHOLDS:
        tp = sum(score >= threshold and label for score, label in scored)
        fp = sum(score >= threshold and not label for score, label in scored)
        fn = len(true_candidates) - tp
        predicted = tp + fp
        precision = tp / predicted if predicted else 0.0
        recall = tp / len(true_candidates) if true_candidates else 0.0
        table.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f0.5": f05(tp, fp, fn),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "predicted": predicted,
            }
        )

    summary: dict[str, float | int] = {
        "sampled_s1": len(s1_records),
        "bounded_candidate_pool_per_source": candidate_pool_size,
        "total_true_matches": len(true_pairs),
        "true_matches_in_candidates": len(true_candidates),
        "blocking_recall": len(true_candidates) / len(true_pairs) if true_pairs else 0.0,
        "true_matches_missed_by_blocking": len(true_pairs) - len(true_candidates),
        "candidate_pairs": len(candidate_pairs),
        "candidate_true_matches_for_p3_metrics": len(true_candidates),
    }
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help="maximum prefix records loaded per candidate source (default: 50000)",
    )
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if args.candidate_pool_size < 1:
        parser.error("--candidate-pool-size must be positive")

    table, summary = evaluate(args.sample_size, args.candidate_pool_size)
    print("\nTraining evaluation summary (bounded P2 candidate pool)")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nthreshold\tprecision\trecall\tF0.5\tTP\tFP\tFN\tpredicted")
    for row in table:
        print(
            f"{row['threshold']:.2f}\t{row['precision']:.4f}\t{row['recall']:.4f}\t"
            f"{row['f0.5']:.4f}\t{row['TP']}\t{row['FP']}\t{row['FN']}\t{row['predicted']}"
        )
    best = max(table, key=lambda row: (row["f0.5"], row["precision"], row["threshold"]))
    print(f"\nBest sampled threshold by F0.5, then precision: {best['threshold']:.2f}")


if __name__ == "__main__":
    main()
