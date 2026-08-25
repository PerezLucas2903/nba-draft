# NBA draft: getting drafted vs. getting minutes

This project answers two related, but deliberately separate, questions:

1. Which post-Combine prospects are most likely to be drafted?
2. Once a player is drafted, what predicts his NBA minutes over the next three seasons?

The entry classifier uses NCAA and NBA Draft Combine information. The minutes
model uses NCAA features and draft capital, with NBA regular-season minutes as
its outcome. Both cover the 2009-2022 draft classes and use chronological
evaluation, with 2020-2022 reserved as the final holdout.

## The two frozen results

| Task | Population | Inputs | 2020-2022 holdout result |
|---|---|---:|---|
| Draft-entry classification | All four-source Combine participants | 105 numerical + 2 categorical | AP 0.8245, ROC-AUC 0.7506 |
| Early-minutes regression | Drafted players | 48 numerical | R² 0.4819, RMSE 1568.2 |

These are different prediction problems. Draft position is forbidden in the
entry classifier because it reveals the outcome. It is valid in the minutes
model because that model is conditional on the player already having been
drafted.

The winning JSON files are kept as frozen reference experiments:

- [Classification config](configs/xgboost_draft_classifier_advanced_best.json)
- [Regression config](configs/xgboost_minutes_advanced_best.json)

## Setup

Python 3.10 or newer is required. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Get the data

Raw and processed data are generated locally and intentionally ignored by Git.
Follow the [data collection guide](nba_draft_data_collection/README.md) to
install the separate collection dependencies, download the public NCAA and NBA
sources, validate the download, and build the processed tables supported by the
tracked scripts.

The two frozen configs require these exact local snapshots:

- `nba_draft_data_collection/data/processed/draft_classification_advanced.parquet`
- `nba_draft_data_collection/data/processed/drafted_players_advanced.parquet`

The repository currently does not include the final materialization step that
created those two advanced snapshots. The collection guide documents the raw
download and the reproducible base/enriched builders, but rerunning the frozen
results also requires placing the advanced snapshots at the paths above.

The collection guide uses its own virtual environment. After following it,
return to the repository root and reactivate the modeling environment:

```powershell
cd ..
.venv\Scripts\Activate.ps1
```

On Linux or macOS, use `source .venv/bin/activate` after `cd ..`. Once the two
advanced files are available, run the complete test suite:

```powershell
python -m unittest discover -s tests -v
```

## Train the winners

Each command reads one plain JSON file, fits one model, prints its metrics, and
saves the fitted artifact under `models/`.

```powershell
# Draft-entry classifier: AP 0.8245 and ROC-AUC 0.7506.
python src/train_classifier.py --config configs/xgboost_draft_classifier_advanced_best.json

# First-three-season minutes: R² 0.4819.
python src/train_model.py --config configs/xgboost_minutes_advanced_best.json
```

The classification artifact is a dictionary containing its fitted pipeline,
sigmoid calibrator, annual decision rule, and feature metadata. The regression
artifact is the fitted scikit-learn pipeline itself.

## Teaching notebooks

The two small notebooks run the same training functions with the fixed winning
hyperparameters. They do not tune models or overwrite artifacts.

- [Train and inspect the classifier](notebooks/evaluate_best_classification_model.ipynb)
- [Train and inspect the regression model](notebooks/evaluate_best_regression_model.ipynb)

Start Jupyter with:

```powershell
python -m jupyter lab
```

Both notebooks are also committed with executed tables and plots, so the
expected result is visible before rerunning them.

## Train or tune another model

The smaller starter configs are useful for ordinary experiments:

```powershell
python src/train_model.py --config configs/regression.json
python src/tune_model.py --config configs/regression.json

python src/train_classifier.py --config configs/classification.json
python src/tune_classifier.py --config configs/classification.json
```

`--n-iter` makes a quick reproducible sample without changing the JSON:

```powershell
python src/tune_model.py --config configs/regression.json --n-iter 2
python src/tune_classifier.py --config configs/classification.json --n-iter 3
```

The advanced configs can also start a new search. Their search model, report,
and candidate-best JSON use names containing `_search`, so a tuning run cannot
replace either frozen winner:

```powershell
python src/tune_model.py --config configs/xgboost_minutes_advanced_best.json --n-iter 5
python src/tune_classifier.py --config configs/xgboost_draft_classifier_advanced_best.json --n-iter 5
```

The regression search uses the configured expanding validation folds and fixed
reference candidates, then minimizes pooled validation RMSE. The classification
search purges repeated-player overlap in each expanding fold and maximizes macro
normalized average precision. Both refit the winner through 2019 before scoring
the 2020-2022 holdout.

To create a lasting new experiment, copy a config, change its feature or search
blocks, and give every output a new name. Do not point `best_config_path` back at
the input config; both tuners reject that mistake.

## What preprocessing happens

The regression path converts numerical values, replaces infinities with missing
values, learns median replacements and standardization on the fitted rows, and
uses the raw minutes target. Its advanced predictors are already materialized
in the configured Parquet file.

The classification path blocks identity and post-draft leakage fields. Numeric
missing values stay missing so XGBoost can learn a missing branch. Categories
use an explicit `missing` level and one-hot encoding. Repeated players are
removed from earlier fitting periods, sigmoid calibration is learned on
2018-2019, and the final annual rule ranks a complete Combine class rather than
scoring an isolated player.

For the full explanation, formulas, code map, and safe modification recipes,
read the [code and feature-engineering guide](docs/code_and_feature_engineering_guide.pdf).
The editable version is [HTML](docs/code_and_feature_engineering_guide.html).

A more formal record of data preparation, validation, calibration, and caveats
is in the [preprocessing and validation report](docs/model_preprocessing_report.pdf),
also available as [HTML](docs/model_preprocessing_report.html).

## Project layout

```text
configs/       JSON contracts for training and tuning
src/           Short CLI scripts and reusable workflow functions
nba_draft_data_collection/  Downloaders, dataset builders, and local data workspace
notebooks/     Analysis plus the two teaching notebooks
docs/          Searchable reports and their editable HTML sources
tests/         Compatibility, leakage, tuning, and result checks
models/        Reproducible local outputs (ignored by git)
```

This is a predictive study, not a causal one. The entry result applies to the
Combine cohort, not every NCAA player. The minutes result applies only after a
player is drafted, and draft capital mixes team evaluation with opportunity.
