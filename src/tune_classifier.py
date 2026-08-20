"""Tune the draft-entry classifier with the search block in a JSON config."""

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
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from . import classifier_workflow as workflow
    from .tuning import parameter_candidates
except ImportError:
    import classifier_workflow as workflow
    from tuning import parameter_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--n-iter",
        type=int,
        help="Temporarily override tuning.n_iter for this run.",
    )
    return parser.parse_args()


def config_with_parameters(
    config: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    candidate = copy.deepcopy(config)
    current = candidate["model"].get("hyperparameters", {})
    candidate["model"]["hyperparameters"] = current | parameters
    return candidate


def protect_source_config(source: Path, best_config_path: Path) -> None:
    """A search result must never replace the frozen input experiment."""
    if source.resolve() == best_config_path.resolve():
        raise ValueError(
            "tuning.best_config_path cannot overwrite the source config; "
            "choose a separate search output"
        )


def validation_score(
    metric: str,
    actual: pd.Series,
    probability: Any,
    years: Any | None = None,
) -> float:
    if actual.nunique() != 2:
        raise ValueError("The validation split must contain both target classes")
    if metric == "average_precision":
        return float(average_precision_score(actual, probability))
    if metric == "roc_auc":
        return float(roc_auc_score(actual, probability))
    if metric != "macro_normalized_average_precision":
        raise ValueError(
            "tuning metric must be average_precision, roc_auc, or "
            "macro_normalized_average_precision"
        )
    if years is None:
        raise ValueError("Macro normalized AP requires validation years")

    frame = pd.DataFrame(
        {
            "actual": np.asarray(actual, dtype=int),
            "probability": np.asarray(probability, dtype=float),
            "year": np.asarray(years),
        }
    )
    annual_scores = []
    for _, group in frame.groupby("year", sort=True):
        if group["actual"].nunique() != 2:
            continue
        prevalence = float(group["actual"].mean())
        ap = float(average_precision_score(group["actual"], group["probability"]))
        annual_scores.append((ap - prevalence) / (1.0 - prevalence))
    if not annual_scores:
        raise ValueError("Macro normalized AP needs a binary validation year")
    return float(np.mean(annual_scores))


def make_validation_folds(
    data: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build the configured expanding folds, stopping before the test years."""
    fold_configs = config["tuning"].get("validation_folds")
    if not fold_configs:
        splits = workflow.chronological_splits(data, config["split"])
        train, purged = workflow.purge_group_overlap(
            splits["train"], splits["validation"], config.get("group_column")
        )
        return [
            {
                "name": "validation",
                "train": train,
                "validation": splits["validation"],
                "purged_training_rows": purged,
            }
        ]

    year_column = config["split"]["column"]
    test_start = int(config["split"]["test_start"])
    target = config["target"]
    used_years: set[int] = set()
    folds = []
    for number, fold_config in enumerate(fold_configs, start=1):
        train_end = int(fold_config["train_end"])
        start = int(fold_config["validation_start"])
        end = int(fold_config["validation_end"])
        if not train_end < start <= end:
            raise ValueError(
                "Each validation fold must satisfy train_end < "
                "validation_start <= validation_end"
            )
        if end >= test_start:
            raise ValueError("Validation folds cannot include final test years")
        years = set(range(start, end + 1))
        if years & used_years:
            raise ValueError("Validation fold year ranges cannot overlap")
        used_years.update(years)

        train = data.loc[data[year_column].le(train_end)].copy()
        validation = data.loc[data[year_column].between(start, end)].copy()
        train, purged = workflow.purge_group_overlap(
            train, validation, config.get("group_column")
        )
        if train.empty or validation.empty:
            raise ValueError(f"Validation fold {number} contains an empty split")
        if train[target].nunique() != 2 or validation[target].nunique() != 2:
            raise ValueError(f"Validation fold {number} must contain both classes")
        folds.append(
            {
                "name": fold_config.get("name", f"fold_{number}"),
                "train": train,
                "validation": validation,
                "purged_training_rows": purged,
            }
        )
    return folds


def evaluate_candidate(
    config: dict[str, Any],
    parameters: dict[str, Any],
    folds: list[dict[str, Any]],
    metric: str,
) -> tuple[float, list[dict[str, Any]]]:
    features = config["numerical_features"] + config["categorical_features"]
    target = config["target"]
    year_column = config["split"]["column"]
    candidate = config_with_parameters(config, parameters)
    actual_parts, probability_parts, year_parts = [], [], []
    fold_reports = []

    for fold in folds:
        model = workflow.make_pipeline(candidate)
        model.fit(fold["train"][features], fold["train"][target])
        probability = model.predict_proba(fold["validation"][features])[:, 1]
        actual_parts.append(fold["validation"][target].to_numpy())
        probability_parts.append(probability)
        year_parts.append(fold["validation"][year_column].to_numpy())
        fold_reports.append(
            {
                "name": fold["name"],
                "purged_training_rows": fold["purged_training_rows"],
                **workflow.probability_metrics(
                    fold["validation"][target], probability
                ),
            }
        )

    actual = pd.Series(np.concatenate(actual_parts))
    probability = np.concatenate(probability_parts)
    years = np.concatenate(year_parts)
    return validation_score(metric, actual, probability, years), fold_reports


def main() -> None:
    args = parse_args()
    source_config = args.config.resolve()
    config = workflow.load_config(source_config)
    tuning = config.get("tuning", {})
    if not tuning.get("search_space"):
        raise ValueError("Add tuning.search_space to the configuration")

    output_config = workflow.resolve(
        tuning.get("best_config_path", "configs/classification_search_best.json")
    )
    protect_source_config(source_config, output_config)

    data = workflow.prepare_data(
        workflow.load_data(workflow.resolve(config["data_path"])), config
    )
    splits = workflow.chronological_splits(data, config["split"])
    folds = make_validation_folds(data, config)
    metric = tuning.get(
        "selection_metric", tuning.get("metric", "average_precision")
    )
    n_iter = args.n_iter if args.n_iter is not None else tuning.get("n_iter")
    random_state = int(tuning.get("random_state", 42))
    candidates = parameter_candidates(
        tuning["search_space"],
        n_iter=n_iter,
        random_state=random_state,
        fixed_candidates=tuning.get("fixed_candidates"),
    )

    print(f"Evaluating {len(candidates)} candidates on {len(folds)} fold(s)...")
    results = []
    best_score = float("-inf")
    best_parameters = None
    for number, parameters in enumerate(candidates, start=1):
        started = perf_counter()
        score, fold_reports = evaluate_candidate(config, parameters, folds, metric)
        results.append(
            {
                "parameters": parameters,
                f"validation_{metric}": score,
                "fold_metrics": fold_reports,
                "fit_seconds": perf_counter() - started,
            }
        )
        if score > best_score:
            best_score = score
            best_parameters = parameters
        print(f"[{number}/{len(candidates)}] {metric}={score:.4f}")

    if best_parameters is None:
        raise RuntimeError("The search produced no candidates")
    best_config = config_with_parameters(config, best_parameters)
    model, calibrator, decision_rule, fit_info = workflow.fit_configured_classifier(
        best_config, splits
    )
    features = config["numerical_features"] + config["categorical_features"]
    target = config["target"]
    final_metrics = {
        name: workflow.evaluate(
            model,
            frame,
            features,
            target,
            calibrator=calibrator,
            decision_rule=decision_rule,
            year_column=config["split"]["column"],
        )
        for name, frame in splits.items()
    }

    model_path = workflow.resolve(
        tuning.get("output_model_path", "models/classifier_search.joblib")
    )
    results_path = workflow.resolve(
        tuning.get("results_path", "models/classifier_search_results.json")
    )
    predictions_path = (
        workflow.resolve(tuning["test_predictions_path"])
        if tuning.get("test_predictions_path")
        else None
    )
    importance_path = (
        workflow.resolve(tuning["feature_importance_path"])
        if tuning.get("feature_importance_path")
        else None
    )
    for path in (model_path, results_path, output_config, predictions_path, importance_path):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    best_config["output_model_path"] = str(model_path)
    best_config["report_path"] = str(results_path)
    best_config["refit_on_train_validation"] = True
    best_config["decision_rule_fitted"] = decision_rule
    best_config["search_run"] = {
        "source_config": source_config.as_posix(),
        "effective_n_iter": n_iter,
        "random_state": random_state,
    }
    if predictions_path is not None:
        best_config["test_predictions_path"] = str(predictions_path)
    if importance_path is not None:
        best_config["feature_importance_path"] = str(importance_path)

    results.sort(key=lambda row: row[f"validation_{metric}"], reverse=True)
    report = {
        "source_config": source_config.as_posix(),
        "search": {
            "effective_n_iter": n_iter,
            "random_state": random_state,
            "candidate_count": len(candidates),
            "fixed_candidates": tuning.get("fixed_candidates", []),
            "search_space": tuning["search_space"],
            "folds": [
                {
                    "name": fold["name"],
                    "train_rows": len(fold["train"]),
                    "validation_rows": len(fold["validation"]),
                    "purged_rows": fold["purged_training_rows"],
                }
                for fold in folds
            ],
        },
        "selection_metric": metric,
        "selection_score": best_score,
        "best_parameters": best_parameters,
        "fit": fit_info,
        "decision_rule": decision_rule,
        "final_metrics": final_metrics,
        "final_metrics_note": config.get("evaluation_note"),
        "candidates": results,
    }

    artifact = workflow.make_artifact(best_config, model, calibrator, decision_rule)
    joblib.dump(artifact, model_path)
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_config.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    if predictions_path is not None:
        workflow.make_test_predictions(
            best_config, model, calibrator, decision_rule, splits["test"]
        ).to_csv(predictions_path, index=False)
    importance = workflow.feature_importance(model)
    if importance_path is not None and importance is not None:
        importance.to_csv(importance_path, index=False)

    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))
    print(f"Saved search model to {model_path}")
    print(f"Saved candidate config to {output_config}")


if __name__ == "__main__":
    main()
