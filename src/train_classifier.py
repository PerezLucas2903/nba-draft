"""Train the draft-entry classifier described by a JSON config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

try:  # Package import in tests/notebooks; local import when run as a script.
    from . import classifier_workflow as workflow
except ImportError:
    import classifier_workflow as workflow


# Keep the useful notebook/test imports available from this familiar module.
resolve = workflow.resolve
feature_leakage_reason = workflow.feature_leakage_reason
load_config = workflow.load_config
load_data = workflow.load_data
prepare_data = workflow.prepare_data
make_model = workflow.make_model
make_pipeline = workflow.make_pipeline
chronological_splits = workflow.chronological_splits
purge_group_overlap = workflow.purge_group_overlap
fit_probability_calibrator = workflow.fit_probability_calibrator
apply_probability_calibrator = workflow.apply_probability_calibrator
predict_annual_top_fraction = workflow.predict_annual_top_fraction
probability_metrics = workflow.probability_metrics
fit_configured_classifier = workflow.fit_configured_classifier
evaluate = workflow.evaluate
make_artifact = workflow.make_artifact
make_test_predictions = workflow.make_test_predictions
feature_importance = workflow.feature_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _path(config: dict, direct_key: str, tuning_key: str, default: str) -> Path:
    """Direct-training paths take priority over search-output paths."""
    value = config.get(direct_key) or config.get("tuning", {}).get(
        tuning_key, default
    )
    return resolve(value)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    data = prepare_data(load_data(resolve(config["data_path"])), config)
    splits = chronological_splits(data, config["split"])
    features = config["numerical_features"] + config["categorical_features"]
    target = config["target"]

    model, calibrator, decision_rule, fit_info = fit_configured_classifier(
        config, splits
    )
    print(
        "Fit rows: "
        f"{fit_info['fit_rows']} "
        f"(purged {fit_info['calibration_training_rows_purged']} before "
        "calibration and "
        f"{fit_info['final_training_rows_purged']} before the final refit)."
    )

    metrics = {
        name: evaluate(
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
    raw_metrics = {
        name: probability_metrics(
            frame[target], model.predict_proba(frame[features])[:, 1]
        )
        for name, frame in splits.items()
    }
    report = {
        "final_metrics": metrics,
        "raw_probability_metrics": raw_metrics,
        "decision_rule": decision_rule,
        "evaluation_note": config.get("evaluation_note"),
        "calibration": {
            "method": config.get("probability_calibration", {}).get(
                "method", "none"
            ),
            "sigmoid_coefficient": (
                float(calibrator.coef_[0, 0]) if calibrator is not None else None
            ),
            "sigmoid_intercept": (
                float(calibrator.intercept_[0]) if calibrator is not None else None
            ),
            "validation_raw": fit_info.get("calibration_validation_raw"),
            "validation_calibrated": fit_info.get(
                "calibration_validation_calibrated"
            ),
        },
        "fit": fit_info,
    }

    model_path = _path(
        config, "output_model_path", "output_model_path", "models/classifier.joblib"
    )
    report_path = _path(
        config, "report_path", "results_path", "models/classifier_metrics.json"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    needs_bundle = calibrator is not None or decision_rule["type"] != "probability_threshold"
    saved_model = (
        make_artifact(config, model, calibrator, decision_rule)
        if needs_bundle
        else model
    )
    joblib.dump(saved_model, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions_value = config.get("test_predictions_path") or config.get(
        "tuning", {}
    ).get("test_predictions_path")
    if predictions_value:
        predictions_path = resolve(predictions_value)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        make_test_predictions(
            config, model, calibrator, decision_rule, splits["test"]
        ).to_csv(predictions_path, index=False)

    importance_value = config.get("feature_importance_path") or config.get(
        "tuning", {}
    ).get("feature_importance_path")
    importance = feature_importance(model)
    if importance_value and importance is not None:
        importance_path = resolve(importance_value)
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(importance_path, index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {report_path}")


if __name__ == "__main__":
    main()
