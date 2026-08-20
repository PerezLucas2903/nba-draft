from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.train_model import (
    evaluate,
    feature_columns,
    load_config,
    load_data,
    make_pipeline,
    prepare_data,
    resolve_path,
    split_data,
)
from src.tune_model import (
    make_validation_folds,
    protect_source_config,
    score_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ADVANCED_CONFIG = REPO_ROOT / "configs" / "xgboost_minutes_advanced_best.json"


class AdvancedRegressionCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ADVANCED_CONFIG)
        raw = load_data(resolve_path(cls.config["data_path"]))
        cls.data = prepare_data(raw, cls.config)
        cls.features = feature_columns(cls.config)

    def test_advanced_expanding_folds_are_restored(self) -> None:
        folds = make_validation_folds(self.data, self.config)

        self.assertEqual(
            [
                (fold["name"], len(fold["train"]), len(fold["validation"]))
                for fold in folds
            ],
            [
                ("2014_2015", 300, 120),
                ("2016_2017", 420, 120),
                ("2018_2019", 540, 120),
            ],
        )

    def test_historical_winner_has_expected_pooled_rmse(self) -> None:
        folds = make_validation_folds(self.data, self.config)
        result = score_candidate(
            self.config,
            self.config["model"]["hyperparameters"],
            folds,
            self.features,
            self.config["target"],
        )

        self.assertAlmostEqual(result["validation_rmse"], 1447.8603104310112)
        self.assertEqual(len(result["fold_metrics"]), 3)

    def test_direct_training_preserves_holdout_r2(self) -> None:
        splits = split_data(self.data, self.config["split"])
        fit_data = pd.concat([splits["train"], splits["validation"]])
        model = make_pipeline(self.config)
        model.fit(
            fit_data[self.features],
            fit_data[self.config["target"]],
        )

        metrics = evaluate(
            model,
            {"test": splits["test"]},
            self.features,
            self.config["target"],
        )

        self.assertAlmostEqual(metrics["test"]["r2"], 0.4819487044423807)

    def test_tuner_refuses_to_replace_its_source_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "source config"):
            protect_source_config(ADVANCED_CONFIG, ADVANCED_CONFIG)


if __name__ == "__main__":
    unittest.main()
