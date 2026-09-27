"""Tests for P4 evaluation utilities."""

import tempfile
import unittest
from pathlib import Path

from src.evaluate import (
    calculate_f_beta,
    blocking_recall,
    compare_thresholds,
    evaluate_pairs,
    evaluate_threshold,
    find_false_negatives,
    find_false_positives,
    load_ground_truth,
    load_predictions,
    prediction_pairs,
    predictions_at_threshold,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        ground_truth = {
            ("S1-1", "S2-1"),
            ("S1-2", "S3-2"),
        }

        predictions = [
            {
                "s1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
            },
            {
                "s1_entity_id": "S1-2",
                "candidate_entity_id": "S3-2",
            },
        ]

        result = evaluate_pairs(
            ground_truth,
            predictions,
        )

        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.false_negatives, 0)

        self.assertAlmostEqual(
            result.precision,
            1.0,
        )

        self.assertAlmostEqual(
            result.recall,
            1.0,
        )

        self.assertAlmostEqual(
            result.f0_5,
            1.0,
        )

    def test_false_positive_and_false_negative(self) -> None:
        ground_truth = {
            ("S1-1", "S2-1"),
            ("S1-2", "S3-2"),
        }

        predictions = [
            {
                "s1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
            },
            {
                "s1_entity_id": "S1-3",
                "candidate_entity_id": "S2-3",
            },
        ]

        result = evaluate_pairs(
            ground_truth,
            predictions,
        )

        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 1)

        self.assertAlmostEqual(
            result.precision,
            0.5,
        )

        self.assertAlmostEqual(
            result.recall,
            0.5,
        )

        self.assertAlmostEqual(
            result.f0_5,
            0.5,
        )

    def test_no_predictions(self) -> None:
        ground_truth = {
            ("S1-1", "S2-1"),
        }

        result = evaluate_pairs(
            ground_truth,
            [],
        )

        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.false_negatives, 1)

        self.assertEqual(
            result.precision,
            0.0,
        )

        self.assertEqual(
            result.recall,
            0.0,
        )

        self.assertEqual(
            result.f0_5,
            0.0,
        )

    def test_duplicate_predictions_are_counted_once(self) -> None:
        ground_truth = {
            ("S1-1", "S2-1"),
        }

        predictions = [
            {
                "s1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
            },
            {
                "s1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
            },
        ]

        result = evaluate_pairs(
            ground_truth,
            predictions,
        )

        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.predicted_pairs, 1)

    def test_f05_formula(self) -> None:
        result = calculate_f_beta(
            precision=0.8,
            recall=0.5,
            beta=0.5,
        )

        expected = (
            1.25 * 0.8 * 0.5
            / (0.25 * 0.8 + 0.5)
        )

        self.assertAlmostEqual(
            result,
            expected,
        )


class BlockingRecallTests(unittest.TestCase):
    def test_blocking_recall(self) -> None:
        ground_truth = {
            ("S1-1", "S2-1"),
            ("S1-2", "S2-2"),
            ("S1-3", "S3-3"),
            ("S1-4", "S3-4"),
        }

        candidates = [
            {
                "s1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
            },
            {
                "s1_entity_id": "S1-2",
                "candidate_entity_id": "S2-2",
            },
            {
                "s1_entity_id": "S1-5",
                "candidate_entity_id": "S2-5",
            },
        ]

        recall = blocking_recall(
            ground_truth,
            candidates,
        )

        self.assertAlmostEqual(
            recall,
            0.5,
        )

    def test_empty_ground_truth(self) -> None:
        recall = blocking_recall(
            set(),
            [],
        )

        self.assertEqual(
            recall,
            0.0,
        )


class ThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ground_truth = {
            ("S1-1", "S2-1"),
            ("S1-2", "S2-2"),
        }

        self.predictions = [
            {
                "s1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
                "score": "0.95",
            },
            {
                "s1_entity_id": "S1-2",
                "candidate_entity_id": "S2-2",
                "score": "0.75",
            },
            {
                "s1_entity_id": "S1-3",
                "candidate_entity_id": "S2-3",
                "score": "0.60",
            },
        ]

    def test_threshold_keeps_high_scores(self) -> None:
        selected = predictions_at_threshold(
            self.predictions,
            0.80,
        )

        self.assertEqual(
            len(selected),
            1,
        )

        self.assertEqual(
            selected[0]["candidate_entity_id"],
            "S2-1",
        )

    def test_threshold_includes_equal_score(self) -> None:
        selected = predictions_at_threshold(
            self.predictions,
            0.75,
        )

        self.assertEqual(
            len(selected),
            2,
        )

    def test_threshold_evaluation(self) -> None:
        result = evaluate_threshold(
            self.ground_truth,
            self.predictions,
            0.70,
        )

        self.assertEqual(
            result.true_positives,
            2,
        )

        self.assertEqual(
            result.false_positives,
            0,
        )

        self.assertEqual(
            result.false_negatives,
            0,
        )

        self.assertAlmostEqual(
            result.f0_5,
            1.0,
        )

    def test_threshold_comparison(self) -> None:
        results = compare_thresholds(
            self.ground_truth,
            self.predictions,
            thresholds=[
                0.50,
                0.70,
                0.90,
            ],
        )

        self.assertEqual(
            len(results),
            3,
        )

        self.assertEqual(
            results[0][0],
            0.50,
        )

        self.assertEqual(
            results[1][0],
            0.70,
        )

        self.assertEqual(
            results[2][0],
            0.90,
        )

        self.assertEqual(
            results[1][1].true_positives,
            2,
        )

        self.assertEqual(
            results[2][1].true_positives,
            1,
        )

    def test_invalid_threshold(self) -> None:
        with self.assertRaises(ValueError):
            predictions_at_threshold(
                self.predictions,
                1.5,
            )


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ground_truth = {
            ("S1-1", "S2-1"),
            ("S1-2", "S2-2"),
            ("S1-3", "S3-3"),
        }

        self.predictions = [
            {
                "s1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
            },
            {
                "s1_entity_id": "S1-2",
                "candidate_entity_id": "S2-WRONG",
            },
            {
                "s1_entity_id": "S1-4",
                "candidate_entity_id": "S2-4",
            },
        ]

    def test_false_positives(self) -> None:
        false_positives = find_false_positives(
            self.ground_truth,
            self.predictions,
        )

        self.assertEqual(
            false_positives,
            {
                ("S1-2", "S2-WRONG"),
                ("S1-4", "S2-4"),
            },
        )

    def test_false_negatives(self) -> None:
        false_negatives = find_false_negatives(
            self.ground_truth,
            self.predictions,
        )

        self.assertEqual(
            false_negatives,
            {
                ("S1-2", "S2-2"),
                ("S1-3", "S3-3"),
            },
        )


class PairUtilityTests(unittest.TestCase):
    def test_prediction_pairs(self) -> None:
        predictions = [
            {
                "s1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
            },
            {
                "s1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
            },
            {
                "s1_entity_id": "S1-2",
                "candidate_entity_id": "S3-2",
            },
        ]

        pairs = prediction_pairs(predictions)

        self.assertEqual(
            pairs,
            {
                ("S1-1", "S2-1"),
                ("S1-2", "S3-2"),
            },
        )


class FileLoadingTests(unittest.TestCase):
    def test_load_ground_truth(self) -> None:
        content = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-1\tS2-1,S3-1\n"
            "S1-2\tS2-2\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ground_truth.tsv"
            path.write_text(
                content,
                encoding="utf-8",
            )

            pairs = load_ground_truth(path)

        self.assertEqual(
            pairs,
            {
                ("S1-1", "S2-1"),
                ("S1-1", "S3-1"),
                ("S1-2", "S2-2"),
            },
        )

    def test_load_predictions(self) -> None:
        content = (
            "s1_entity_id\tcandidate_entity_id\tscore\n"
            "S1-1\tS2-1\t0.95\n"
            "S1-2\tS3-2\t0.75\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.tsv"
            path.write_text(
                content,
                encoding="utf-8",
            )

            predictions = load_predictions(path)

        self.assertEqual(
            len(predictions),
            2,
        )

        self.assertEqual(
            predictions[0]["s1_entity_id"],
            "S1-1",
        )

        self.assertEqual(
            predictions[0]["candidate_entity_id"],
            "S2-1",
        )

        self.assertEqual(
            predictions[0]["score"],
            "0.95",
        )

    def test_invalid_ground_truth_columns(self) -> None:
        content = "wrong_column\tanother_column\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.tsv"
            path.write_text(
                content,
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_ground_truth(path)

    def test_invalid_prediction_columns(self) -> None:
        content = "wrong_column\tanother_column\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.tsv"
            path.write_text(
                content,
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_predictions(path)


if __name__ == "__main__":
    unittest.main()