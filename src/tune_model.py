"""Tune a regression model with the search space in a JSON config."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

try:
    from .train_model import (
        evaluate,
        feature_columns,
        load_config,
        load_data,
        make_pipeline,
        prepare_data,
        resolve_path,
        split_data,
    )
    from .tuning import parameter_candidates
except ImportError:  # Allows ``python src/tune_model.py``.
    from train_model import (
        evaluate,
        feature_columns,
        load_config,
        load_data,
        make_pipeline,
        prepare_data,
        resolve_path,
        split_data,
    )
    from tuning import parameter_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--n-iter",
        type=int,
        help="Override tuning.n_iter. Use fewer iterations for a quick run.",
    )
    return parser.parse_args()


def config_with_parameters(
    config: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    candidate = copy.deepcopy(config)
    current = candidate["model"].get("hyperparameters", {})
    candidate["model"]["hyperparameters"] = current | parameters
    return candidate


def make_validation_folds(
    data: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build the configured expanding windows, or use the main validation split."""
    fold_configs = config["tuning"].get("validation_folds")
    if not fold_configs:
        splits = split_data(data, config["split"])
        return [
            {
                "name": "validation",
                "train": splits["train"],
                "validation": splits["validation"],
            }
        ]

    year_column = config["split"]["column"]
    test_start = int(config["split"]["test_start"])
    used_validation_years: set[int] = set()
    folds = []

    for index, fold_config in enumerate(fold_configs, start=1):
        train_end = int(fold_config["train_end"])
        validation_start = int(fold_config["validation_start"])
        validation_end = int(fold_config["validation_end"])

        if not train_end < validation_start <= validation_end:
            raise ValueError(
                "Each validation fold must satisfy train_end < "
                "validation_start <= validation_end"
            )
        if validation_end >= test_start:
            raise ValueError("Validation folds cannot include final test years")

        validation_years = set(range(validation_start, validation_end + 1))
        if used_validation_years & validation_years:
            raise ValueError("Validation fold year ranges cannot overlap")
        used_validation_years.update(validation_years)

        train = data.loc[data[year_column].le(train_end)].copy()
        validation = data.loc[
            data[year_column].between(validation_start, validation_end)
        ].copy()
        if train.empty or validation.empty:
            raise ValueError(f"Validation fold {index} contains an empty split")

        folds.append(
            {
                "name": fold_config.get("name", f"fold_{index}"),
                "train": train,
                "validation": validation,
            }
        )

    return folds


def score_candidate(
    config: dict[str, Any],
    parameters: dict[str, Any],
    folds: list[dict[str, Any]],
    features: list[str],
    target: str,
) -> dict[str, Any]:
    """Fit one parameter set and pool its out-of-fold predictions."""
    started = perf_counter()
    candidate_config = config_with_parameters(config, parameters)
    actual_parts = []
    predicted_parts = []
    fold_metrics = []

    for fold in folds:
        model = make_pipeline(candidate_config)
        train = fold["train"]
        validation = fold["validation"]
        model.fit(train[features], train[target])
        predicted = np.asarray(model.predict(validation[features]))
        actual = validation[target].to_numpy()

        actual_parts.append(actual)
        predicted_parts.append(predicted)
        fold_mse = float(mean_squared_error(actual, predicted))
        fold_metrics.append(
            {
                "name": fold["name"],
                "train_rows": len(train),
                "validation_rows": len(validation),
                "rmse": fold_mse**0.5,
                "r2": float(r2_score(actual, predicted)),
            }
        )

    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(predicted_parts)
    pooled_mse = float(mean_squared_error(actual, predicted))
    fold_rmses = [fold["rmse"] for fold in fold_metrics]
    return {
        "parameters": parameters,
        "validation_rmse": pooled_mse**0.5,
        "validation_r2": float(r2_score(actual, predicted)),
        "validation_rmse_fold_mean": float(np.mean(fold_rmses)),
        "validation_rmse_fold_std": float(np.std(fold_rmses)),
        "fold_metrics": fold_metrics,
        "fit_seconds": perf_counter() - started,
    }


def protect_source_config(source: Path, best_config_path: Path) -> None:
    """Refuse a tuning run that would replace the config it was launched from."""
    if source.resolve() == best_config_path.resolve():
        raise ValueError(
            "tuning.best_config_path resolves to the source config. "
            "Choose a different output path so the frozen config is not overwritten."
        )


def main() -> None:
    args = parse_args()
    source_config = args.config.resolve()
    config = load_config(source_config)
    tuning = config.get("tuning")
    if not tuning or not tuning.get("search_space"):
        raise ValueError("Add tuning.search_space to the configuration")

    model_path_value = tuning.get(
        "output_model_path", "models/regression_tuned.joblib"
    )
    output_model = resolve_path(model_path_value)
    output_results = resolve_path(
        tuning.get("results_path", "models/regression_tuning_results.json")
    )
    output_config = resolve_path(
        tuning.get("best_config_path", "configs/regression_best.json")
    )
    protect_source_config(source_config, output_config)

    features = feature_columns(config)
    target = config["target"]
    data = load_data(resolve_path(config["data_path"]))
    data = prepare_data(data, config)
    splits = split_data(data, config["split"])
    folds = make_validation_folds(data, config)

    n_iter = args.n_iter if args.n_iter is not None else tuning.get("n_iter")
    random_state = int(tuning.get("random_state", 42))
    candidates = parameter_candidates(
        tuning["search_space"],
        n_iter=n_iter,
        random_state=random_state,
        fixed_candidates=tuning.get("fixed_candidates"),
    )
    print(f"Evaluating {len(candidates)} hyperparameter combinations...")

    results = []
    for index, parameters in enumerate(candidates, start=1):
        result = score_candidate(config, parameters, folds, features, target)
        results.append(result)
        print(
            f"[{index}/{len(candidates)}] validation "
            f"RMSE={result['validation_rmse']:.4f}"
        )

    if not results:
        raise RuntimeError("The hyperparameter search produced no candidates")

    results.sort(key=lambda item: item["validation_rmse"])
    best_result = results[0]
    best_parameters = best_result["parameters"]
    best_rmse = best_result["validation_rmse"]
    best_config = config_with_parameters(config, best_parameters)
    train_validation = pd.concat([splits["train"], splits["validation"]])
    final_model = make_pipeline(best_config)
    final_model.fit(train_validation[features], train_validation[target])
    final_metrics = evaluate(final_model, splits, features, target)

    for path in (output_model, output_results, output_config):
        path.parent.mkdir(parents=True, exist_ok=True)

    best_config["output_model_path"] = model_path_value
    best_config["refit_on_train_validation"] = True
    report = {
        "source_config": args.config.as_posix(),
        "search": {
            "model": config["model"]["name"],
            "n_iter": n_iter,
            "configured_n_iter": tuning.get("n_iter"),
            "cli_n_iter_override": args.n_iter,
            "random_state": random_state,
            "candidate_count": len(candidates),
            "search_space": tuning["search_space"],
            "fixed_candidates": tuning.get("fixed_candidates", []),
            "validation_folds": [
                {
                    "name": fold["name"],
                    "training_rows": len(fold["train"]),
                    "validation_rows": len(fold["validation"]),
                }
                for fold in folds
            ],
        },
        "best_parameters": best_parameters,
        "selection_validation_rmse": best_rmse,
        "selection_method": (
            "pooled expanding-window out-of-fold RMSE"
            if tuning.get("validation_folds")
            else "single chronological validation RMSE"
        ),
        "final_metrics": final_metrics,
        "final_metrics_note": (
            "Parameters were selected on validation. After refitting on train + "
            "validation, the reported train and validation metrics are in-sample; "
            "test is the untouched chronological holdout."
        ),
        "candidates": results,
    }

    joblib.dump(final_model, output_model)
    output_results.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_config.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    summary = report | {"candidates": f"{len(results)} saved to {output_results}"}
    print(json.dumps(summary, indent=2))
    print(f"Saved tuned model to {output_model}")
    print(f"Saved best configuration to {output_config}")


if __name__ == "__main__":
    main()
