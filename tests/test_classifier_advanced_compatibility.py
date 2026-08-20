from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_classifier import (
    apply_probability_calibrator,
    chronological_splits,
    fit_probability_calibrator,
    load_config,
    load_data,
    make_pipeline,
    predict_annual_top_fraction,
    prepare_data,
    probability_metrics,
    purge_group_overlap,
    resolve,
)
from src.tune_classifier import (
    make_validation_folds,
    protect_source_config,
    validation_score,
)
from src.tuning import parameter_candidates


REPO_ROOT = Path(__file__).resolve().parents[1]
ADVANCED_CONFIG = (
    REPO_ROOT / "configs" / "xgboost_draft_classifier_advanced_best.json"
)


def tiny_xgboost_config(
    *, numerical: list[str], categorical: list[str]
) -> dict:
    return {
        "data_path": "unused.csv",
        "target": "drafted",
        "numerical_features": numerical,
        "categorical_features": categorical,
        "numerical_missing_strategy": "native",
        "split": {
            "column": "draft_year",
            "train_end": 2012,
            "validation_start": 2013,
            "validation_end": 2013,
            "test_start": 2014,
            "test_end": 2014,
        },
        "model": {
            "name": "xgboost",
            "hyperparameters": {
                "n_estimators": 5,
                "max_depth": 1,
                "learning_rate": 0.1,
                "random_state": 7,
                "n_jobs": 1,
                "verbosity": 0,
            },
        },
    }


class AdvancedConfigIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ADVANCED_CONFIG)

    def test_promoted_advanced_config_loads_with_compatibility_options(self) -> None:
        self.assertEqual(len(self.config["numerical_features"]), 105)
        self.assertEqual(
            self.config["categorical_features"],
            ["position", "college_position"],
        )
        self.assertEqual(self.config["numerical_missing_strategy"], "native")
        self.assertEqual(
            self.config["probability_calibration"]["method"], "sigmoid"
        )
        self.assertEqual(
            self.config["tuning"]["decision_rule"], "annual_top_fraction"
        )
        self.assertAlmostEqual(
            self.config["decision_rule_fitted"]["fraction"],
            0.6301369863013698,
        )
        self.assertEqual(
            self.config["model"]["hyperparameters"]["scale_pos_weight"],
            0.3,
        )

    def test_tuner_refuses_to_replace_the_frozen_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "source config"):
            protect_source_config(ADVANCED_CONFIG, ADVANCED_CONFIG)

    def test_advanced_dataset_split_and_group_purge_counts(self) -> None:
        data = load_data(resolve(self.config["data_path"]))
        prepared = prepare_data(data, self.config)
        splits = chronological_splits(prepared, self.config["split"])

        self.assertEqual(
            {name: len(frame) for name, frame in splits.items()},
            {"train": 510, "validation": 146, "test": 202},
        )

        group_column = self.config["group_column"]
        calibration_train, calibration_purged = purge_group_overlap(
            splits["train"], splits["validation"], group_column
        )
        final_train, final_purged = purge_group_overlap(
            pd.concat([splits["train"], splits["validation"]]),
            splits["test"],
            group_column,
        )

        self.assertEqual(calibration_purged, 6)
        self.assertEqual(final_purged, 8)
        self.assertEqual(len(calibration_train), 504)
        self.assertEqual(len(final_train), 648)
        self.assertTrue(
            set(calibration_train[group_column].dropna()).isdisjoint(
                set(splits["validation"][group_column].dropna())
            )
        )
        self.assertTrue(
            set(final_train[group_column].dropna()).isdisjoint(
                set(splits["test"][group_column].dropna())
            )
        )


class AdvancedPreprocessingTests(unittest.TestCase):
    def test_pd_na_and_unseen_category_fit_and_predict(self) -> None:
        config = tiny_xgboost_config(
            numerical=["height", "speed"], categorical=["position"]
        )
        raw = pd.DataFrame(
            {
                "height": [72.0, np.nan, 80.0, 76.0, np.nan, 77.0],
                "speed": [3.2, 3.4, np.nan, 3.3, 3.1, np.nan],
                "position": pd.Series(
                    ["G", "F", "C", pd.NA, "G-F", pd.NA], dtype="string"
                ),
                "drafted": [0, 0, 1, 1, 0, 1],
                "draft_year": [2011, 2011, 2012, 2012, 2013, 2013],
            }
        )
        prepared = prepare_data(raw, config)
        features = config["numerical_features"] + config["categorical_features"]
        train = prepared.iloc[:4]
        future = prepared.iloc[4:]

        self.assertEqual(prepared["position"].dtype, object)
        self.assertTrue(pd.isna(prepared.loc[3, "position"]))

        model = make_pipeline(config)
        model.fit(train[features], train["drafted"])
        probabilities = model.predict_proba(future[features])[:, 1]

        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())

    def test_native_numeric_missing_values_reach_xgboost(self) -> None:
        config = tiny_xgboost_config(
            numerical=["height", "speed"], categorical=[]
        )
        frame = pd.DataFrame(
            {
                "height": [72.0, np.nan, 80.0, 76.0],
                "speed": [3.2, 3.4, np.nan, 3.3],
                "drafted": [0, 0, 1, 1],
            }
        )
        features = config["numerical_features"]
        model = make_pipeline(config)

        self.assertEqual(
            model.named_steps["preprocessor"].transformers[0][1],
            "passthrough",
        )
        model.fit(frame[features], frame["drafted"])
        transformed = model.named_steps["preprocessor"].transform(frame[features])
        probabilities = model.predict_proba(frame[features])[:, 1]

        self.assertTrue(np.isnan(np.asarray(transformed, dtype=float)).any())
        self.assertTrue(np.isfinite(probabilities).all())


class CalibrationAndDecisionRuleTests(unittest.TestCase):
    def test_sigmoid_calibration_is_finite_monotone_and_optional(self) -> None:
        actual = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        raw = np.array(
            [0.45, 0.46, 0.47, 0.48, 0.49, 0.51, 0.52, 0.53, 0.54, 0.55]
        )
        calibrator = fit_probability_calibrator(
            actual,
            raw,
            {"method": "sigmoid", "C": 100.0, "random_state": 7},
        )

        calibrated = apply_probability_calibrator(raw, calibrator)
        boundary_mapping = apply_probability_calibrator(
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]), calibrator
        )

        self.assertTrue(np.isfinite(boundary_mapping).all())
        self.assertTrue(((boundary_mapping >= 0) & (boundary_mapping <= 1)).all())
        self.assertTrue((np.diff(boundary_mapping) >= 0).all())
        self.assertFalse(np.allclose(calibrated, raw))
        self.assertLess(
            probability_metrics(actual, calibrated)["brier_score"],
            probability_metrics(actual, raw)["brier_score"],
        )

        no_calibrator = fit_probability_calibrator(
            actual, raw, {"method": "none"}
        )
        self.assertIsNone(no_calibrator)
        np.testing.assert_array_equal(
            apply_probability_calibrator(raw, no_calibrator), raw
        )

    def test_annual_top_fraction_groups_by_year_and_breaks_ties_stably(self) -> None:
        probabilities = np.array([0.10, 0.80, 0.90, 0.20, 0.70])
        years = np.array([2020, 2021, 2020, 2021, 2021])

        selected = predict_annual_top_fraction(probabilities, years, 0.5)
        transformed = predict_annual_top_fraction(
            np.exp(probabilities), years, 0.5
        )

        np.testing.assert_array_equal(selected, np.array([0, 1, 1, 0, 1]))
        np.testing.assert_array_equal(transformed, selected)
        np.testing.assert_array_equal(
            predict_annual_top_fraction(
                np.array([0.8, 0.8, 0.1]),
                np.array([2020, 2020, 2020]),
                1.0 / 3.0,
            ),
            np.array([1, 0, 0]),
        )


class AdvancedTuningCompatibilityTests(unittest.TestCase):
    def test_fixed_candidates_are_kept_first_and_deduplicated(self) -> None:
        candidates = parameter_candidates(
            {"max_depth": [1, 2]},
            n_iter=None,
            fixed_candidates=[
                {"max_depth": 1},
                {"max_depth": 3},
                {"max_depth": 3},
            ],
        )

        self.assertEqual(
            candidates,
            [
                {"max_depth": 1},
                {"max_depth": 3},
                {"max_depth": 2},
            ],
        )

    def test_expanding_folds_purge_groups_and_stop_before_test(self) -> None:
        data = pd.DataFrame(
            {
                "player_id": ["a", "b", "c", "d", "a", "e", "c", "f", "g", "h"],
                "draft_year": [2010, 2010, 2011, 2011, 2012, 2012, 2013, 2013, 2014, 2014],
                "drafted": [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
            }
        )
        config = {
            "target": "drafted",
            "group_column": "player_id",
            "split": {
                "column": "draft_year",
                "train_end": 2012,
                "validation_start": 2013,
                "validation_end": 2013,
                "test_start": 2014,
                "test_end": 2014,
            },
            "tuning": {
                "validation_folds": [
                    {
                        "name": "2012",
                        "train_end": 2011,
                        "validation_start": 2012,
                        "validation_end": 2012,
                    },
                    {
                        "name": "2013",
                        "train_end": 2012,
                        "validation_start": 2013,
                        "validation_end": 2013,
                    },
                ]
            },
        }

        folds = make_validation_folds(data, config)

        self.assertEqual([fold["name"] for fold in folds], ["2012", "2013"])
        self.assertEqual(
            [fold["purged_training_rows"] for fold in folds], [1, 1]
        )
        self.assertLess(len(folds[0]["train"]), len(folds[1]["train"]))
        for fold in folds:
            self.assertLess(
                fold["train"]["draft_year"].max(),
                fold["validation"]["draft_year"].min(),
            )
            self.assertTrue(
                set(fold["train"]["player_id"]).isdisjoint(
                    set(fold["validation"]["player_id"])
                )
            )

        invalid = copy.deepcopy(config)
        invalid["tuning"]["validation_folds"] = [
            {
                "train_end": 2013,
                "validation_start": 2014,
                "validation_end": 2014,
            }
        ]
        with self.assertRaisesRegex(ValueError, "final test years"):
            make_validation_folds(data, invalid)

    def test_macro_normalized_average_precision_is_averaged_by_year(self) -> None:
        actual = pd.Series([0, 1, 0, 0, 1, 1])
        probability = np.array([0.1, 0.9, 0.9, 0.8, 0.7, 0.6])
        years = np.array([2020, 2020, 2021, 2021, 2021, 2021])

        score = validation_score(
            "macro_normalized_average_precision",
            actual,
            probability,
            years,
        )

        self.assertAlmostEqual(score, 5.0 / 12.0)


if __name__ == "__main__":
    unittest.main()
