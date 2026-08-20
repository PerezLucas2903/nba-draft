import unittest

import pandas as pd

from src import tune_classifier, tune_model
from src.train_classifier import chronological_splits, purge_group_overlap
from src.train_model import prepare_data as prepare_regression_data
from src.train_model import split_data
from src.tuning import parameter_candidates


class ParameterCandidateTests(unittest.TestCase):
    def test_exhaustive_grid(self):
        candidates = parameter_candidates(
            {
                "learning_rate": [0.1, 0.2],
                "max_depth": [2, 3],
            },
            n_iter=None,
        )

        self.assertEqual(
            candidates,
            [
                {"learning_rate": 0.1, "max_depth": 2},
                {"learning_rate": 0.1, "max_depth": 3},
                {"learning_rate": 0.2, "max_depth": 2},
                {"learning_rate": 0.2, "max_depth": 3},
            ],
        )

    def test_random_sample_is_deterministic_and_capped(self):
        search_space = {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        }

        first = parameter_candidates(search_space, n_iter=3, random_state=7)
        second = parameter_candidates(search_space, n_iter=3, random_state=7)
        capped = parameter_candidates(search_space, n_iter=100, random_state=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(capped), 9)
        self.assertCountEqual(
            capped,
            parameter_candidates(search_space, n_iter=None),
        )

    def test_invalid_iteration_count_is_rejected(self):
        for n_iter in (0, -1):
            with self.subTest(n_iter=n_iter):
                with self.assertRaisesRegex(ValueError, "at least 1"):
                    parameter_candidates({"max_depth": [2, 3]}, n_iter=n_iter)

    def test_empty_search_space_or_parameter_values_are_rejected(self):
        invalid_spaces = ({}, {"max_depth": []})

        for search_space in invalid_spaces:
            with self.subTest(search_space=search_space):
                with self.assertRaises(ValueError):
                    parameter_candidates(search_space, n_iter=None)


class ChronologicalSplitTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"year": [2017, 2018, 2019, 2020]})
        self.invalid_split = {
            "column": "year",
            "train_end": 2018,
            "validation_start": 2018,
            "validation_end": 2019,
            "test_start": 2020,
            "test_end": 2020,
        }

    def test_regression_split_rejects_overlapping_periods(self):
        with self.assertRaisesRegex(ValueError, "train_end < validation_start"):
            split_data(self.data, self.invalid_split)

    def test_classifier_split_rejects_overlapping_periods(self):
        with self.assertRaisesRegex(ValueError, "train_end < validation_start"):
            chronological_splits(self.data, self.invalid_split)


class DataSafetyTests(unittest.TestCase):
    def test_group_overlap_is_removed_from_training(self):
        training = pd.DataFrame(
            {"player_id": ["a", "b", "c"], "drafted": [0, 1, 0]}
        )
        later = pd.DataFrame({"player_id": ["a", "d"], "drafted": [1, 0]})

        purged, count = purge_group_overlap(training, later, "player_id")

        self.assertEqual(count, 1)
        self.assertEqual(purged["player_id"].tolist(), ["b", "c"])

    def test_regression_csv_values_are_coerced_to_numeric(self):
        config = {
            "target": "minutes",
            "numerical_features": ["points"],
            "categorical_features": [],
            "log_transform_target": False,
            "split": {"column": "year"},
        }
        data = pd.DataFrame(
            {
                "minutes": ["100", "200"],
                "points": ["5.5", "bad"],
                "year": ["2019", "2020"],
            }
        )

        prepared = prepare_regression_data(data, config)

        self.assertEqual(prepared["minutes"].tolist(), [100, 200])
        self.assertEqual(prepared["year"].tolist(), [2019, 2020])
        self.assertEqual(prepared["points"].iloc[0], 5.5)
        self.assertTrue(pd.isna(prepared["points"].iloc[1]))


class ConfigMergeTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": {
                "name": "xgboost",
                "hyperparameters": {
                    "max_depth": 2,
                    "learning_rate": 0.1,
                },
            },
            "numerical_features": ["points"],
        }

    def assert_merged_without_mutation(self, merge):
        candidate = merge(
            self.config,
            {"max_depth": 4, "n_estimators": 200},
        )

        self.assertEqual(
            candidate["model"]["hyperparameters"],
            {
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_estimators": 200,
            },
        )
        self.assertEqual(
            self.config["model"]["hyperparameters"],
            {"max_depth": 2, "learning_rate": 0.1},
        )
        self.assertIsNot(candidate["model"], self.config["model"])
        self.assertIsNot(
            candidate["numerical_features"],
            self.config["numerical_features"],
        )

    def test_regression_config_merge_does_not_mutate_original(self):
        self.assert_merged_without_mutation(tune_model.config_with_parameters)

    def test_classifier_config_merge_does_not_mutate_original(self):
        self.assert_merged_without_mutation(
            tune_classifier.config_with_parameters
        )


class ClassifierValidationScoreTests(unittest.TestCase):
    def test_supported_metrics(self):
        actual = pd.Series([0, 1, 0, 1])
        probability = [0.1, 0.9, 0.2, 0.8]

        self.assertAlmostEqual(
            tune_classifier.validation_score(
                "average_precision", actual, probability
            ),
            1.0,
        )
        self.assertAlmostEqual(
            tune_classifier.validation_score("roc_auc", actual, probability),
            1.0,
        )

    def test_invalid_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "average_precision.*roc_auc"):
            tune_classifier.validation_score(
                "accuracy",
                pd.Series([0, 1]),
                [0.2, 0.8],
            )

    def test_one_class_validation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "both target classes"):
            tune_classifier.validation_score(
                "average_precision",
                pd.Series([1, 1]),
                [0.7, 0.8],
            )


if __name__ == "__main__":
    unittest.main()
