"""Small, reusable pieces for the draft-entry classifier.

The command-line scripts stay short by keeping data preparation, calibration,
and prediction rules here.  Everything is still plain functions and dictionaries
so the same steps are easy to use from a notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]

# These fields either identify a person or reveal something that happens only
# after the draft.  They are useful for auditing rows, but never as predictors.
FORBIDDEN_FEATURES = {
    "drafted",
    "overall_pick",
    "round_number",
    "round_pick",
    "team_abbreviation",
    "draft_team",
    "draft_organization",
    "organization",
    "organization_type",
    "draft_year",
    "nba_player_id",
    "player_id",
    "temp_player_id",
    "player_name",
    "first_name",
    "last_name",
    "college_espn_id",
    "athlete_id",
    "normalized_name",
    "identity_match_method",
    "college_match_method",
    "player_profile_flag",
    "combine_available",
    "has_college_data",
    "nba_min_3y",
}


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def feature_leakage_reason(column: str) -> str | None:
    name = column.lower()
    if name in FORBIDDEN_FEATURES or name.startswith("nba_"):
        return "known outcome, identity, or post-draft field"
    if name == "id" or name.endswith("_id"):
        return "identifier"
    if name == "name" or name.endswith("_name"):
        return "name"
    if "match_method" in name:
        return "identity-match metadata"
    if name in {"season", "year"} or name.endswith(("_season", "_year")):
        return "absolute calendar field"
    return None


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "data_path",
        "target",
        "numerical_features",
        "categorical_features",
        "split",
        "model",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    numerical = config["numerical_features"]
    categorical = config["categorical_features"]
    if not isinstance(numerical, list) or not isinstance(categorical, list):
        raise ValueError("numerical_features and categorical_features must be lists")
    features = numerical + categorical
    if not features:
        raise ValueError("Configure at least one feature")
    if len(features) != len(set(features)):
        raise ValueError("Feature lists contain duplicates")
    if config["target"] in features:
        raise ValueError("The target cannot also be a feature")

    unsafe = sorted(name for name in features if feature_leakage_reason(name))
    if unsafe:
        raise ValueError(f"Outcome or identity fields cannot be features: {unsafe}")
    if config.get("numerical_missing_strategy", "median") not in {
        "median",
        "native",
    }:
        raise ValueError("numerical_missing_strategy must be median or native")
    if config.get("probability_calibration", {}).get("method", "none") not in {
        "none",
        "sigmoid",
    }:
        raise ValueError("probability_calibration.method must be none or sigmoid")

    fitted_rule = config.get("decision_rule_fitted", {})
    rule_type = fitted_rule.get(
        "type",
        config.get("tuning", {}).get(
            "decision_rule", "probability_threshold"
        ),
    )
    if rule_type not in {"probability_threshold", "annual_top_fraction"}:
        raise ValueError(
            "decision rule must be probability_threshold or annual_top_fraction"
        )
    return config


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("data_path must point to a .parquet or .csv file")


def prepare_data(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Check the modeling grain and normalize values before splitting."""
    target = config["target"]
    year_column = config["split"]["column"]
    features = config["numerical_features"] + config["categorical_features"]
    required = list(
        dict.fromkeys(
            features
            + [target, year_column]
            + config.get("id_columns", [])
            + config.get("entity_columns", [])
            + ([config["group_column"]] if config.get("group_column") else [])
        )
    )
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    prepared = data.dropna(subset=[target, year_column]).copy()
    prepared[target] = pd.to_numeric(prepared[target], errors="raise").astype("int8")
    if not set(prepared[target].unique()) <= {0, 1}:
        raise ValueError("Classification target must contain only 0 and 1")
    prepared[year_column] = pd.to_numeric(
        prepared[year_column], errors="raise"
    ).astype(int)

    for column in config["numerical_features"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)
    for column in config["categorical_features"]:
        prepared[column] = prepared[column].astype("object")
        prepared[column] = prepared[column].where(prepared[column].notna(), np.nan)

    entity_columns = config.get("entity_columns", [])
    if entity_columns:
        duplicate = prepared.duplicated(entity_columns, keep=False)
        if duplicate.any():
            sample = prepared.loc[duplicate, entity_columns].head(5)
            raise ValueError(
                f"Expected one row per {entity_columns}; duplicates include "
                f"{sample.to_dict('records')}"
            )
    return prepared


def make_model(config: dict[str, Any]):
    name = config["model"]["name"].lower()
    parameters = config["model"].get("hyperparameters", {})
    if name in {"logistic", "logistic_regression"}:
        return LogisticRegression(**({"max_iter": 2000} | parameters))
    if name in {"xgboost", "xgb"}:
        from xgboost import XGBClassifier

        defaults = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        return XGBClassifier(**(defaults | parameters))
    raise ValueError("model.name must be logistic or xgboost")


def make_pipeline(config: dict[str, Any]) -> Pipeline:
    numerical = config["numerical_features"]
    categorical = config["categorical_features"]
    transformers = []

    if numerical:
        if config.get("numerical_missing_strategy", "median") == "native":
            numeric_steps: Pipeline | str = "passthrough"
        else:
            steps = [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=config.get(
                            "numerical_missing_indicators", False
                        ),
                    ),
                )
            ]
            if config["model"]["name"].lower().startswith("logistic"):
                steps.append(("scale", StandardScaler()))
            numeric_steps = Pipeline(steps)
        transformers.append(("numeric", numeric_steps, numerical))

    if categorical:
        category_steps = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
                ),
            ]
        )
        transformers.append(("categorical", category_steps, categorical))

    preprocessor = ColumnTransformer(transformers)
    return Pipeline([("preprocessor", preprocessor), ("model", make_model(config))])


def chronological_splits(
    data: pd.DataFrame, split: dict[str, int]
) -> dict[str, pd.DataFrame]:
    valid_order = (
        split["train_end"]
        < split["validation_start"]
        <= split["validation_end"]
        < split["test_start"]
        <= split["test_end"]
    )
    if not valid_order:
        raise ValueError(
            "Split values must satisfy train_end < validation_start <= "
            "validation_end < test_start <= test_end"
        )

    year = data[split["column"]]
    frames = {
        "train": data.loc[year.le(split["train_end"])].copy(),
        "validation": data.loc[
            year.between(split["validation_start"], split["validation_end"])
        ].copy(),
        "test": data.loc[
            year.between(split["test_start"], split["test_end"])
        ].copy(),
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError("One or more chronological splits are empty")
    return frames


def purge_group_overlap(
    training: pd.DataFrame,
    later: pd.DataFrame,
    group_column: str | None,
) -> tuple[pd.DataFrame, int]:
    """Remove an earlier player row when that player appears in a later split."""
    if not group_column:
        return training, 0
    overlap = training[group_column].isin(set(later[group_column].dropna()))
    return training.loc[~overlap].copy(), int(overlap.sum())


def _logits(probabilities: Any) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Cannot calibrate non-finite probabilities")
    epsilon = np.finfo(float).eps
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def fit_probability_calibrator(
    actual: Any,
    probabilities: Any,
    settings: dict[str, Any] | None,
) -> LogisticRegression | None:
    settings = settings or {"method": "none"}
    if settings.get("method", "none") == "none":
        return None
    labels = np.asarray(actual, dtype=int)
    if np.unique(labels).size != 2:
        raise ValueError("Probability calibration requires both target classes")
    calibrator = LogisticRegression(
        C=float(settings.get("C", 1.0)),
        solver="lbfgs",
        random_state=int(settings.get("random_state", 42)),
    )
    calibrator.fit(_logits(probabilities).reshape(-1, 1), labels)
    return calibrator


def apply_probability_calibrator(
    probabilities: Any, calibrator: LogisticRegression | None
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if calibrator is None:
        return scores
    return calibrator.predict_proba(_logits(scores).reshape(-1, 1))[:, 1]


def predict_annual_top_fraction(
    probabilities: Any, years: Any, fraction: float
) -> np.ndarray:
    """Select the same score fraction separately inside each draft year."""
    scores = np.asarray(probabilities, dtype=float)
    year_values = np.asarray(years)
    if scores.ndim != 1 or year_values.ndim != 1 or len(scores) != len(year_values):
        raise ValueError("Probabilities and years must be equal-length vectors")
    if not np.isfinite(scores).all() or pd.isna(year_values).any():
        raise ValueError("Annual ranking requires finite scores and complete years")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Annual top fraction must be between 0 and 1")

    predicted = np.zeros(len(scores), dtype=int)
    for year in np.unique(year_values):
        indices = np.flatnonzero(year_values == year)
        count = int(np.floor(fraction * len(indices) + 0.5))
        order = np.argsort(-scores[indices], kind="mergesort")
        predicted[indices[order[:count]]] = 1
    return predicted


def probability_metrics(actual: Any, probability: Any) -> dict[str, float | None]:
    labels = np.asarray(actual, dtype=int)
    scores = np.asarray(probability, dtype=float)
    auc = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": auc,
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, scores)),
    }


def fit_configured_classifier(
    config: dict[str, Any], splits: dict[str, pd.DataFrame]
) -> tuple[Pipeline, LogisticRegression | None, dict[str, Any], dict[str, Any]]:
    """Fit validation calibration first, then refit the final estimator."""
    features = config["numerical_features"] + config["categorical_features"]
    target = config["target"]
    group = config.get("group_column")
    calibration_settings = config.get("probability_calibration")
    fitted_rule = config.get("decision_rule_fitted")
    tuning = config.get("tuning", {})
    rule_type = (fitted_rule or {}).get(
        "type", tuning.get("decision_rule", "probability_threshold")
    )

    # Calibration is learned from a train-only model scored on 2018-2019.
    calibrator = None
    calibration_purged = 0
    raw_validation = None
    calibrated_validation = None
    needs_validation_fit = (
        config.get("probability_calibration", {}).get("method", "none") != "none"
        or (rule_type == "annual_top_fraction" and not fitted_rule)
    )
    if needs_validation_fit:
        calibration_train, calibration_purged = purge_group_overlap(
            splits["train"], splits["validation"], group
        )
        calibration_model = make_pipeline(config)
        calibration_model.fit(calibration_train[features], calibration_train[target])
        raw_validation = calibration_model.predict_proba(
            splits["validation"][features]
        )[:, 1]
        calibrator = fit_probability_calibrator(
            splits["validation"][target], raw_validation, calibration_settings
        )
        calibrated_validation = apply_probability_calibrator(
            raw_validation, calibrator
        )

    if fitted_rule:
        decision_rule = dict(fitted_rule)
    elif rule_type == "annual_top_fraction":
        decision_rule = {
            "type": "annual_top_fraction",
            "fraction": float(splits["validation"][target].mean()),
            "selection_strategy": "calibration_prevalence",
        }
    else:
        decision_rule = {"type": "probability_threshold", "threshold": 0.5}

    refit = config.get("refit_on_train_validation", True)
    if refit:
        fit_data = pd.concat([splits["train"], splits["validation"]])
        later_data = splits["test"]
    else:
        fit_data = splits["train"]
        later_data = pd.concat([splits["validation"], splits["test"]])
    fit_data, final_purged = purge_group_overlap(fit_data, later_data, group)

    model = make_pipeline(config)
    model.fit(fit_data[features], fit_data[target])
    details = {
        "refit_on_train_validation": refit,
        "fit_rows": len(fit_data),
        "calibration_training_rows_purged": calibration_purged,
        "final_training_rows_purged": final_purged,
    }
    if raw_validation is not None:
        details["calibration_validation_raw"] = probability_metrics(
            splits["validation"][target], raw_validation
        )
        details["calibration_validation_calibrated"] = probability_metrics(
            splits["validation"][target], calibrated_validation
        )
    return model, calibrator, decision_rule, details


def evaluate(
    model: Pipeline,
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    calibrator: LogisticRegression | None = None,
    decision_rule: dict[str, Any] | None = None,
    year_column: str | None = None,
) -> dict[str, Any]:
    actual = frame[target].to_numpy()
    raw_probability = model.predict_proba(frame[features])[:, 1]
    probability = apply_probability_calibrator(raw_probability, calibrator)
    rule = decision_rule or {"type": "probability_threshold", "threshold": 0.5}
    if rule["type"] == "annual_top_fraction":
        predicted = predict_annual_top_fraction(
            probability, frame[year_column], float(rule["fraction"])
        )
    else:
        predicted = (probability >= float(rule.get("threshold", 0.5))).astype(int)

    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "rows": len(frame),
        "positive_rate": float(actual.mean()),
        **probability_metrics(actual, probability),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def make_artifact(
    config: dict[str, Any],
    model: Pipeline,
    calibrator: LogisticRegression | None,
    decision_rule: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_version": 3,
        "estimator": model,
        "probability_calibrator": calibrator,
        "decision_threshold": decision_rule.get("threshold"),
        "decision_rule": decision_rule,
        "decision_scope": (
            "complete combine-participant batch for each draft year"
            if decision_rule["type"] == "annual_top_fraction"
            else "independent prospect rows"
        ),
        "target": config["target"],
        "feature_columns": config["numerical_features"]
        + config["categorical_features"],
        "numerical_features": config["numerical_features"],
        "categorical_features": config["categorical_features"],
        "numerical_missing_strategy": config.get(
            "numerical_missing_strategy", "median"
        ),
        "split_column": config["split"]["column"],
        "trained_through_year": int(config["split"]["validation_end"]),
    }


def make_test_predictions(
    config: dict[str, Any],
    model: Pipeline,
    calibrator: LogisticRegression | None,
    decision_rule: dict[str, Any],
    test: pd.DataFrame,
) -> pd.DataFrame:
    features = config["numerical_features"] + config["categorical_features"]
    target = config["target"]
    year_column = config["split"]["column"]
    raw_probability = model.predict_proba(test[features])[:, 1]
    probability = apply_probability_calibrator(raw_probability, calibrator)
    if decision_rule["type"] == "annual_top_fraction":
        predicted = predict_annual_top_fraction(
            probability, test[year_column], float(decision_rule["fraction"])
        )
    else:
        predicted = (
            probability >= float(decision_rule.get("threshold", 0.5))
        ).astype(int)

    columns = list(
        dict.fromkeys(config.get("id_columns", []) + [year_column, target])
    )
    predictions = test[columns].copy()
    predictions["raw_model_score"] = raw_probability
    predictions["draft_probability"] = probability
    predictions["predicted_drafted"] = predicted.astype("int8")
    predictions["probability_rank_within_year"] = (
        predictions.groupby(year_column)["draft_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return predictions


def feature_importance(model: Pipeline) -> pd.DataFrame | None:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame(
        {"feature": feature_names, "importance": estimator.feature_importances_}
    ).sort_values("importance", ascending=False)
