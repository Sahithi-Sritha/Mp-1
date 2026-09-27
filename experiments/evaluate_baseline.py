from __future__ import annotations

import csv
import sys
from pathlib import Path

# ============================================================
# PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (
    build_core_name_index,
    build_name_index,
)

from src.normalize import (
    normalize_business_name,
    normalize_business_name_core,
    normalize_country,
)


# ============================================================
# DATASET PATH
# ============================================================

DATASET_ROOT = (
    Path(r"D:\amazon-ml-school-dataset")
    / "6ab10eb3b23ba_student_resource"
    / "student_resource"
    / "dataset"
)

TRAIN_DIR = DATASET_ROOT / "train"

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# TSV READER
# ============================================================

def read_tsv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        for row in reader:
            yield row


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(path: Path) -> set[tuple[str, str]]:
    ground_truth: set[tuple[str, str]] = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        for row in reader:

            source1_id = row["source1_entity_id"]
            matched_ids = row["matched_entity_ids"]

            if not matched_ids:
                continue

            for matched_id in matched_ids.split(","):

                matched_id = matched_id.strip()

                if matched_id:
                    ground_truth.add(
                        (source1_id, matched_id)
                    )

    return ground_truth


# ============================================================
# EXPERIMENT 1: EXACT NORMALIZED NAME
# ============================================================

def generate_exact_candidates(
    source1_records,
    source2_records,
    source3_records,
):

    print()
    print("Building exact-name indexes...")

    source2_index = build_name_index(
        source2_records
    )

    source3_index = build_name_index(
        source3_records
    )

    candidates: set[tuple[str, str]] = set()

    processed = 0

    for record in source1_records:

        processed += 1

        s1_id = record["entity_id"]

        normalized_name = normalize_business_name(
            record["business_name"]
        )

        if not normalized_name:
            continue

        for s2_id in source2_index.get(
            normalized_name,
            [],
        ):
            candidates.add(
                (s1_id, s2_id)
            )

        for s3_id in source3_index.get(
            normalized_name,
            [],
        ):
            candidates.add(
                (s1_id, s3_id)
            )

        if processed % 100_000 == 0:

            print(
                f"Processed S1 records: "
                f"{processed:,}"
            )

    print(
        f"Total S1 records processed: "
        f"{processed:,}"
    )

    return candidates


# ============================================================
# EXPERIMENT 2: CORE NORMALIZED NAME
# ============================================================

def generate_core_candidates(
    source1_records,
    source2_records,
    source3_records,
):

    print()
    print("Building core-name indexes...")

    source2_index = build_core_name_index(
        source2_records
    )

    source3_index = build_core_name_index(
        source3_records
    )

    candidates: set[tuple[str, str]] = set()

    processed = 0

    for record in source1_records:

        processed += 1

        s1_id = record["entity_id"]

        normalized_core = normalize_business_name_core(
            record["business_name"]
        )

        if not normalized_core:
            continue

        for s2_id in source2_index.get(
            normalized_core,
            [],
        ):
            candidates.add(
                (s1_id, s2_id)
            )

        for s3_id in source3_index.get(
            normalized_core,
            [],
        ):
            candidates.add(
                (s1_id, s3_id)
            )

        if processed % 100_000 == 0:

            print(
                f"Processed S1 records: "
                f"{processed:,}"
            )

    print(
        f"Total S1 records processed: "
        f"{processed:,}"
    )

    return candidates


# ============================================================
# EXPERIMENT 3: CORE NAME + COUNTRY
# ============================================================

def generate_name_country_candidates(
    source1_records,
    source2_records,
    source3_records,
):

    print()
    print("Building name + country indexes...")

    source2_index: dict[
        tuple[str, str],
        list[str],
    ] = {}

    source3_index: dict[
        tuple[str, str],
        list[str],
    ] = {}

    # --------------------------------------------------------
    # SOURCE 2
    # --------------------------------------------------------

    print("Indexing train_source2.tsv...")

    for record in source2_records:

        normalized_core = normalize_business_name_core(
            record["business_name"]
        )

        country = normalize_country(
            record["country"]
        )

        if not normalized_core or not country:
            continue

        key = (
            normalized_core,
            country,
        )

        source2_index.setdefault(
            key,
            [],
        ).append(
            record["entity_id"]
        )

    print(
        f"Unique S2 name+country keys: "
        f"{len(source2_index):,}"
    )

    # --------------------------------------------------------
    # SOURCE 3
    # --------------------------------------------------------

    print("Indexing train_source3.tsv...")

    for record in source3_records:

        normalized_core = normalize_business_name_core(
            record["business_name"]
        )

        country = normalize_country(
            record["country"]
        )

        if not normalized_core or not country:
            continue

        key = (
            normalized_core,
            country,
        )

        source3_index.setdefault(
            key,
            [],
        ).append(
            record["entity_id"]
        )

    print(
        f"Unique S3 name+country keys: "
        f"{len(source3_index):,}"
    )

    # --------------------------------------------------------
    # GENERATE CANDIDATES
    # --------------------------------------------------------

    candidates: set[tuple[str, str]] = set()

    processed = 0

    for record in source1_records:

        processed += 1

        s1_id = record["entity_id"]

        normalized_core = normalize_business_name_core(
            record["business_name"]
        )

        country = normalize_country(
            record["country"]
        )

        if not normalized_core or not country:
            continue

        key = (
            normalized_core,
            country,
        )

        for s2_id in source2_index.get(
            key,
            [],
        ):
            candidates.add(
                (s1_id, s2_id)
            )

        for s3_id in source3_index.get(
            key,
            [],
        ):
            candidates.add(
                (s1_id, s3_id)
            )

        if processed % 100_000 == 0:

            print(
                f"Processed S1 records: "
                f"{processed:,}"
            )

    print(
        f"Total S1 records processed: "
        f"{processed:,}"
    )

    return candidates


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    name: str,
    candidates: set[tuple[str, str]],
    ground_truth: set[tuple[str, str]],
):

    true_pairs_retained = len(
        candidates & ground_truth
    )

    if ground_truth:

        recall = (
            true_pairs_retained
            / len(ground_truth)
        )

    else:

        recall = 0.0

    print()
    print(
        f"========== {name} =========="
    )

    print(
        f"Ground-truth pairs : "
        f"{len(ground_truth):,}"
    )

    print(
        f"Candidate pairs    : "
        f"{len(candidates):,}"
    )

    print(
        f"True pairs retained: "
        f"{true_pairs_retained:,}"
    )

    print(
        f"Blocking recall    : "
        f"{recall:.4%}"
    )

    print(
        "============================="
    )

    return {
        "candidates": len(candidates),
        "true_pairs": true_pairs_retained,
        "recall": recall,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("       BLOCKING EXPERIMENT")
    print("==========================================")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print()
    print("Loading training data...")

    source1 = list(
        read_tsv(SOURCE1_PATH)
    )

    source2 = list(
        read_tsv(SOURCE2_PATH)
    )

    source3 = list(
        read_tsv(SOURCE3_PATH)
    )

    print(
        f"S1 records: {len(source1):,}"
    )

    print(
        f"S2 records: {len(source2):,}"
    )

    print(
        f"S3 records: {len(source3):,}"
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    print()
    print("Loading ground truth...")

    ground_truth = load_ground_truth(
        GROUND_TRUTH_PATH
    )

    print(
        f"Ground-truth pairs: "
        f"{len(ground_truth):,}"
    )

    # ========================================================
    # 1. EXACT NAME
    # ========================================================

    print()
    print("Generating exact-name candidates...")

    exact_candidates = generate_exact_candidates(
        source1,
        source2,
        source3,
    )

    exact_results = evaluate(
        "EXACT NORMALIZED NAME",
        exact_candidates,
        ground_truth,
    )

    # ========================================================
    # 2. CORE NAME
    # ========================================================

    print()
    print("Generating core-name candidates...")

    core_candidates = generate_core_candidates(
        source1,
        source2,
        source3,
    )

    core_results = evaluate(
        "CORE NORMALIZED NAME",
        core_candidates,
        ground_truth,
    )

    # ========================================================
    # 3. CORE NAME + COUNTRY
    # ========================================================

    print()
    print(
        "Generating core-name + country candidates..."
    )

    name_country_candidates = (
        generate_name_country_candidates(
            source1,
            source2,
            source3,
        )
    )

    name_country_results = evaluate(
        "CORE NAME + COUNTRY",
        name_country_candidates,
        ground_truth,
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("========== COMPARISON ==========")

    print(
        f"Exact name recall          : "
        f"{exact_results['recall']:.4%}"
    )

    print(
        f"Core name recall           : "
        f"{core_results['recall']:.4%}"
    )

    print(
        f"Core name + country recall : "
        f"{name_country_results['recall']:.4%}"
    )

    print()

    print(
        f"Exact candidates           : "
        f"{exact_results['candidates']:,}"
    )

    print(
        f"Core candidates            : "
        f"{core_results['candidates']:,}"
    )

    print(
        f"Name + country candidates  : "
        f"{name_country_results['candidates']:,}"
    )

    print()

    print(
        f"Core additional recall     : "
        f"{(
            core_results['recall']
            - exact_results['recall']
        ):.4%}"
    )

    print(
        f"Country additional recall  : "
        f"{(
            name_country_results['recall']
            - exact_results['recall']
        ):.4%}"
    )

    print()

    print(
        f"Candidates removed by "
        f"country blocking          : "
        f"{(
            core_results['candidates']
            - name_country_results['candidates']
        ):,}"
    )

    print("================================")


if __name__ == "__main__":
    main()