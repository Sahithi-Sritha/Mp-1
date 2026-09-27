"""Evaluation utilities for business entity resolution.

P4 responsibilities:
- Load ground-truth S1 -> S2/S3 matches.
- Evaluate predicted candidate pairs.
- Calculate precision, recall, F-beta and F0.5.
- Compare score thresholds.
- Report false positives and false negatives.
- Measure blocking recall independently from model performance.

Expected prediction columns:
    s1_entity_id
    candidate_entity_id
    score              (optional for pair-level evaluation,
                        required for threshold evaluation)

Ground-truth columns:
    source1_entity_id
    matched_entity_ids
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


GROUND_TRUTH_S1_COLUMN = "source1_entity_id"
GROUND_TRUTH_MATCHES_COLUMN = "matched_entity_ids"

PREDICTION_S1_COLUMN = "s1_entity_id"
PREDICTION_CANDIDATE_COLUMN = "candidate_entity_id"
PREDICTION_SCORE_COLUMN = "score"


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics for a set of predicted entity pairs."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f0_5: float

    @property
    def predicted_pairs(self) -> int:
        """Number of predicted pairs."""
        return self.true_positives + self.false_positives

    @property
    def ground_truth_pairs(self) -> int:
        """Number of ground-truth pairs."""
        return (
            self.true_positives
            + self.false_negatives
        )


def _pair(
    s1_entity_id: str,
    candidate_entity_id: str,
) -> tuple[str, str]:
    """Create a canonical pair representation."""
    return (
        str(s1_entity_id).strip(),
        str(candidate_entity_id).strip(),
    )


def load_ground_truth(
    path: str | Path,
) -> set[tuple[str, str]]:
    """Load ground-truth S1 -> S2/S3 pairs from a TSV file.

    The ground-truth format is:

        source1_entity_id    matched_entity_ids

    where matched_entity_ids is a comma-separated list.
    """
    path = Path(path)

    pairs: set[tuple[str, str]] = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file, delimiter="\t")

        required = {
            GROUND_TRUTH_S1_COLUMN,
            GROUND_TRUTH_MATCHES_COLUMN,
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Ground-truth file must contain columns: "
                f"{sorted(required)}"
            )

        for row in reader:
            s1_id = (row.get(GROUND_TRUTH_S1_COLUMN) or "").strip()
            matched_ids = (
                row.get(GROUND_TRUTH_MATCHES_COLUMN) or ""
            ).strip()

            if not s1_id or not matched_ids:
                continue

            for candidate_id in matched_ids.split(","):
                candidate_id = candidate_id.strip()

                if candidate_id:
                    pairs.add(_pair(s1_id, candidate_id))

    return pairs


def load_predictions(
    path: str | Path,
) -> list[dict[str, str]]:
    """Load prediction/candidate pairs from a TSV file.

    Required columns:
        s1_entity_id
        candidate_entity_id

    Optional column:
        score
    """
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file, delimiter="\t")

        required = {
            PREDICTION_S1_COLUMN,
            PREDICTION_CANDIDATE_COLUMN,
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Prediction file must contain columns: "
                f"{sorted(required)}"
            )

        predictions: list[dict[str, str]] = []

        for row in reader:
            s1_id = (
                row.get(PREDICTION_S1_COLUMN) or ""
            ).strip()

            candidate_id = (
                row.get(PREDICTION_CANDIDATE_COLUMN) or ""
            ).strip()

            if not s1_id or not candidate_id:
                continue

            prediction = {
                PREDICTION_S1_COLUMN: s1_id,
                PREDICTION_CANDIDATE_COLUMN: candidate_id,
            }

            if PREDICTION_SCORE_COLUMN in row:
                prediction[PREDICTION_SCORE_COLUMN] = (
                    row.get(PREDICTION_SCORE_COLUMN) or ""
                ).strip()

            predictions.append(prediction)

    return predictions


def prediction_pairs(
    predictions: Iterable[Mapping[str, str]],
) -> set[tuple[str, str]]:
    """Convert prediction records into unique entity pairs."""
    pairs: set[tuple[str, str]] = set()

    for prediction in predictions:
        s1_id = prediction.get(PREDICTION_S1_COLUMN, "")
        candidate_id = prediction.get(
            PREDICTION_CANDIDATE_COLUMN,
            "",
        )

        if not s1_id or not candidate_id:
            continue

        pairs.add(_pair(s1_id, candidate_id))

    return pairs


def calculate_f_beta(
    precision: float,
    recall: float,
    beta: float,
) -> float:
    """Calculate F-beta.

    For F0.5, precision receives more weight than recall.
    """
    if precision < 0 or recall < 0:
        raise ValueError("Precision and recall cannot be negative.")

    if beta <= 0:
        raise ValueError("Beta must be greater than zero.")

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta * beta

    denominator = (
        beta_squared * precision + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta_squared)
        * precision
        * recall
        / denominator
    )


def evaluate_pairs(
    ground_truth: set[tuple[str, str]],
    predictions: Iterable[Mapping[str, str]],
) -> EvaluationResult:
    """Evaluate predicted pairs against ground truth."""
    predicted = prediction_pairs(predictions)

    true_positives = len(
        predicted.intersection(ground_truth)
    )

    false_positives = len(
        predicted - ground_truth
    )

    false_negatives = len(
        ground_truth - predicted
    )

    predicted_count = len(predicted)
    ground_truth_count = len(ground_truth)

    if predicted_count:
        precision = true_positives / predicted_count
    else:
        precision = 0.0

    if ground_truth_count:
        recall = true_positives / ground_truth_count
    else:
        recall = 0.0

    f0_5 = calculate_f_beta(
        precision,
        recall,
        beta=0.5,
    )

    return EvaluationResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f0_5=f0_5,
    )


def blocking_recall(
    ground_truth: set[tuple[str, str]],
    candidate_pairs: Iterable[Mapping[str, str]],
) -> float:
    """Calculate recall of a candidate/blocking stage.

    This measures how many true pairs survived blocking before
    the ML matcher is applied.
    """
    if not ground_truth:
        return 0.0

    candidates = prediction_pairs(candidate_pairs)

    retained = len(
        ground_truth.intersection(candidates)
    )

    return retained / len(ground_truth)


def predictions_at_threshold(
    predictions: Iterable[Mapping[str, str]],
    threshold: float,
) -> list[dict[str, str]]:
    """Keep predictions whose score is >= threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    selected: list[dict[str, str]] = []

    for prediction in predictions:
        raw_score = prediction.get(
            PREDICTION_SCORE_COLUMN
        )

        if raw_score is None or raw_score == "":
            raise ValueError(
                "Score is required for threshold evaluation."
            )

        try:
            score = float(raw_score)
        except ValueError as exc:
            raise ValueError(
                f"Invalid prediction score: {raw_score!r}"
            ) from exc

        if not 0 <= score <= 1:
            raise ValueError(
                f"Prediction score must be between 0 and 1: {score}"
            )

        if score >= threshold:
            selected.append(dict(prediction))

    return selected


def evaluate_threshold(
    ground_truth: set[tuple[str, str]],
    predictions: Iterable[Mapping[str, str]],
    threshold: float,
) -> EvaluationResult:
    """Evaluate predictions after applying a score threshold."""
    selected = predictions_at_threshold(
        predictions,
        threshold,
    )

    return evaluate_pairs(
        ground_truth,
        selected,
    )


def compare_thresholds(
    ground_truth: set[tuple[str, str]],
    predictions: Iterable[Mapping[str, str]],
    thresholds: Iterable[float] = (
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ),
) -> list[tuple[float, EvaluationResult]]:
    """Evaluate several score thresholds."""
    prediction_list = list(predictions)

    results: list[tuple[float, EvaluationResult]] = []

    for threshold in thresholds:
        result = evaluate_threshold(
            ground_truth,
            prediction_list,
            threshold,
        )

        results.append((threshold, result))

    return results


def find_false_positives(
    ground_truth: set[tuple[str, str]],
    predictions: Iterable[Mapping[str, str]],
) -> set[tuple[str, str]]:
    """Return predicted pairs that are not in ground truth."""
    predicted = prediction_pairs(predictions)

    return predicted - ground_truth


def find_false_negatives(
    ground_truth: set[tuple[str, str]],
    predictions: Iterable[Mapping[str, str]],
) -> set[tuple[str, str]]:
    """Return ground-truth pairs that were not predicted."""
    predicted = prediction_pairs(predictions)

    return ground_truth - predicted


def print_result(
    result: EvaluationResult,
    title: str = "EVALUATION",
) -> None:
    """Print an evaluation result in a readable format."""
    print()
    print("=" * 45)
    print(title)
    print("=" * 45)

    print(
        f"Ground-truth pairs : {result.ground_truth_pairs:,}"
    )
    print(
        f"Predicted pairs    : {result.predicted_pairs:,}"
    )
    print(
        f"True positives     : {result.true_positives:,}"
    )
    print(
        f"False positives    : {result.false_positives:,}"
    )
    print(
        f"False negatives    : {result.false_negatives:,}"
    )

    print(
        f"Precision          : {result.precision:.4%}"
    )
    print(
        f"Recall             : {result.recall:.4%}"
    )
    print(
        f"F0.5               : {result.f0_5:.4%}"
    )

    print("=" * 45)


def print_threshold_comparison(
    results: Iterable[tuple[float, EvaluationResult]],
) -> None:
    """Print threshold comparison results."""
    print()
    print("=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F0.5':<14}"
        f"{'Predicted':<14}"
    )

    print("-" * 70)

    for threshold, result in results:
        print(
            f"{threshold:<12.2f}"
            f"{result.precision:<14.4%}"
            f"{result.recall:<14.4%}"
            f"{result.f0_5:<14.4%}"
            f"{result.predicted_pairs:<14,}"
        )

    print("=" * 70)


def evaluate_prediction_file(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
) -> EvaluationResult:
    """Convenience function for evaluating a prediction TSV."""
    ground_truth = load_ground_truth(
        ground_truth_path
    )

    predictions = load_predictions(
        prediction_path
    )

    return evaluate_pairs(
        ground_truth,
        predictions,
    )


def main() -> None:
    """Run evaluation from the command line.

    Usage:
        python src/evaluate.py ground_truth.tsv predictions.tsv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate entity-resolution predictions."
    )

    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Path to train_ground_truth.tsv",
    )

    parser.add_argument(
        "predictions",
        type=Path,
        help="Path to prediction TSV.",
    )

    args = parser.parse_args()

    ground_truth = load_ground_truth(
        args.ground_truth
    )

    predictions = load_predictions(
        args.predictions
    )

    result = evaluate_pairs(
        ground_truth,
        predictions,
    )

    print_result(result)

    if predictions and all(
        PREDICTION_SCORE_COLUMN in prediction
        and prediction[PREDICTION_SCORE_COLUMN] != ""
        for prediction in predictions
    ):
        threshold_results = compare_thresholds(
            ground_truth,
            predictions,
        )

        print_threshold_comparison(
            threshold_results
        )


if __name__ == "__main__":
    main()