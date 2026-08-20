"""Train a Ridge or XGBoost model from a JSON configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)

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
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")

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
        raise ValueError("The target cannot also be used as a feature")
    return config


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def feature_columns(config: dict[str, Any]) -> list[str]:
    """Return model inputs in the same order used by the preprocessing pipeline."""
    return config["numerical_features"] + config["categorical_features"]


def load_data(path: Path) -> pd.DataFrame:
    readers = {
        ".parquet": pd.read_parquet,
        ".csv": pd.read_csv,
    }
    try:
        reader = readers[path.suffix.lower()]
    except KeyError as error:
        raise ValueError("data_path must point to a .parquet or .csv file") from error
    return reader(path)


def prepare_data(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Validate configured columns and make CSV values model-ready."""
    target = config["target"]
    split_column = config["split"]["column"]
    required = feature_columns(config) + [target, split_column]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Columns not found in the data: {missing}")

    prepared = data.dropna(subset=[target, split_column]).copy()
    prepared[target] = pd.to_numeric(prepared[target], errors="raise")
    prepared[split_column] = pd.to_numeric(prepared[split_column], errors="raise")
    for column in config["numerical_features"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)
    uses_log_target = config.get("log_transform_target", True)
    if uses_log_target and prepared[target].le(-1).any():
        raise ValueError(
            "A log-transformed target must contain values greater than -1"
        )
    return prepared


def build_estimator(config: dict[str, Any]):
    model_config = config["model"]
    name = model_config["name"].lower()
    parameters = model_config.get("hyperparameters", {})

    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**parameters)
    if name in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError(
                "XGBoost was selected, but it is not installed. Run: pip install xgboost"
            ) from error
        defaults = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        return XGBRegressor(**(defaults | parameters))
    raise ValueError("model.name must be either 'ridge' or 'xgboost'")


def split_data(
    data: pd.DataFrame, split_config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    train_end = split_config["train_end"]
    validation_start = split_config["validation_start"]
    validation_end = split_config["validation_end"]
    test_start = split_config["test_start"]
    test_end = split_config["test_end"]
    if not train_end < validation_start <= validation_end < test_start <= test_end:
        raise ValueError(
            "Split values must satisfy train_end < validation_start <= "
            "validation_end < test_start <= test_end"
        )

    column = split_config["column"]
    ranges = {
        "train": (None, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, test_end),
    }

    splits = {}
    for name, (start, end) in ranges.items():
        mask = data[column].le(end)
        if start is not None:
            mask &= data[column].ge(start)
        splits[name] = data.loc[mask].copy()
        if splits[name].empty:
            raise ValueError(f"The {name} split is empty")
    return splits


def make_pipeline(config: dict[str, Any]) -> TransformedTargetRegressor | Pipeline:
    numerical = config["numerical_features"]
    categorical = config["categorical_features"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="missing"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", build_estimator(config))]
    )

    if config.get("log_transform_target", True):
        return TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1
        )
    return pipeline


def evaluate(model, splits, feature_columns, target: str) -> dict[str, Any]:
    metrics = {}
    for name, frame in splits.items():
        actual = frame[target]
        predicted = model.predict(frame[feature_columns])
        mse = float(mean_squared_error(actual, predicted))
        metrics[name] = {
            "rows": len(frame),
            "mse": mse,
            "rmse": mse**0.5,
            "r2": float(r2_score(actual, predicted)),
        }
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    target = config["target"]
    features = feature_columns(config)

    data = load_data(resolve_path(config["data_path"]))
    before = len(data)
    data = prepare_data(data, config)
    print(
        f"Loaded {before} rows; retained {len(data)} after dropping rows with "
        "a missing target or split value. Feature gaps are imputed."
    )

    splits = split_data(data, config["split"])
    model = make_pipeline(config)
    fit_data = splits["train"]
    if config.get("refit_on_train_validation", False):
        fit_data = pd.concat([splits["train"], splits["validation"]])
        print(
            "Refitting on train + validation; reported train and validation "
            "metrics are therefore in-sample."
        )
    model.fit(fit_data[features], fit_data[target])

    metrics = evaluate(model, splits, features, target)
    print(json.dumps(metrics, indent=2))

    output_path = resolve_path(config.get("output_model_path", "models/model.joblib"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved fitted pipeline to {output_path}")


if __name__ == "__main__":
    main()
