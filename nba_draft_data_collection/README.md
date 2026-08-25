# NBA Draft data collection

This directory downloads the public NCAA and NBA data used by the project and
builds the processed tables supported by the tracked scripts. All files under
`data/raw/` and `data/processed/` are generated locally and ignored by Git.

No API key or account is required. The collectors need network access to the
public endpoints used by `hoopR` and `nba_api`.

## What is downloaded

For each NCAA men's basketball season, `collect_college.R` saves:

- `player_box`: game-level player box scores;
- `player_season_from_box`: season totals and rates derived from those boxes;
- `player_core`: player identity and biographical fields; and
- `team_stats`: season-level team context.

`collect_crosswalks.R` saves college and NBA identity crosswalks. In particular,
the NBA crosswalk connects ESPN athlete IDs in the college data to NBA.com
player IDs.

For each requested NBA draft year, `collect_nba.py` saves official draft
history and four Combine sources: anthropometrics, drills, spot shooting, and
non-stationary shooting. It also downloads regular-season Base and Advanced
player totals. A successful NBA request is stored as normalized Parquet and as
the original gzipped JSON response.

## Year conventions and project coverage

- An NCAA year is the year in which the season ends. For example, `2022` means
  the season ending in spring 2022.
- Draft years are ordinary calendar years. The modeling cohort is 2009-2022.
- NBA season arguments use the season's starting year. `--nba-season-end 2024`
  therefore includes the 2024-25 season.
- The feature builders can look back as far as `draft_year - 6`. To reproduce
  the project's full college history for the 2009 draft class, collect NCAA
  seasons from 2003 onward.

The project download therefore uses NCAA seasons 2003-2022, draft and Combine
years 2009-2022, and NBA seasons 2009-10 through 2024-25. The extra NBA seasons
provide three complete regular seasons after the 2022 draft.

## Prerequisites

- Python 3.10 or newer;
- R with `Rscript` available on `PATH`;
- the Python and R packages installed below; and
- enough time and disk space for a multi-year download.

Start from the repository root. On Windows PowerShell:

```powershell
cd nba_draft_data_collection
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Rscript install_r_packages.R
```

On Linux or macOS:

```bash
cd nba_draft_data_collection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
Rscript install_r_packages.R
```

The college collector checks for `hoopR >= 3.1.0`, `arrow`, and `dplyr` before
downloading. If it reports an old or missing package, install or update all
three with:

```bash
Rscript -e "install.packages(c('hoopR', 'arrow', 'dplyr'), repos=c('https://sportsdataverse.r-universe.dev', 'https://cloud.r-project.org'))"
```

All remaining commands assume the current directory is
`nba_draft_data_collection`. This matters because the scripts and their default
`data/` paths are relative to the current directory.

## Download the complete project data

Run the three collectors in order:

```bash
Rscript collect_college.R 2003 2022 data/raw/college
Rscript collect_crosswalks.R 2003 2022 2024 data/raw/crosswalks
python collect_nba.py --draft-start 2009 --draft-end 2022 --nba-season-start 2009 --nba-season-end 2024 --output data/raw/nba
```

The NBA collector deliberately waits two to four seconds after successful
requests and retries transient failures, so the complete download is
long-running.

### Simpler one-command download

The orchestrator runs all three collectors:

```bash
python collect_all.py --draft-start 2009 --draft-end 2022
```

This convenient command collects NCAA seasons only from 2009 through 2022. Use
the three commands above when reproducing the full six-season college lookback.
`collect_all.py` accepts `--data-root` to change the output root and `--force`
to replace existing files.

## Run one source at a time

The R scripts use positional arguments.

```bash
# NCAA: start year, end year, output directory, optional force flag
Rscript collect_college.R 2003 2022 data/raw/college
Rscript collect_college.R 2003 2022 data/raw/college true

# Crosswalks: college start, college end, NBA end, output directory, force
Rscript collect_crosswalks.R 2003 2022 2024 data/raw/crosswalks true
```

The NBA script exposes named options. For example:

```bash
python collect_nba.py --draft-start 2009 --draft-end 2022 --nba-season-start 2009 --nba-season-end 2024 --output data/raw/nba
```

If the Combine shooting endpoints are temporarily unavailable, collect draft
history, anthropometrics, drills, and NBA totals first:

```bash
python collect_nba.py --draft-start 2009 --draft-end 2022 --nba-season-start 2009 --nba-season-end 2024 --output data/raw/nba --skip-combine-shooting
```

Run the normal command later to fill the shooting files; already-downloaded
Parquets will be skipped. Both shooting sources must be complete before running
the four-source classification builder. For persistent NBA timeouts or
throttling, increase the pause with options such as
`--sleep-min 5 --sleep-max 10`.

## Output layout

```text
data/
├── raw/
│   ├── college/
│   │   ├── player_box/
│   │   ├── player_core/
│   │   ├── player_season_from_box/
│   │   └── team_stats/
│   ├── crosswalks/
│   │   ├── mbb_player_crosswalk.parquet
│   │   └── nba_player_crosswalk.parquet
│   └── nba/
│       ├── draft/
│       ├── combine/
│       │   ├── anthro/
│       │   ├── drills/
│       │   ├── spot_shooting/
│       │   └── nonstationary_shooting/
│       ├── player_stats/
│       │   ├── base/
│       │   └── advanced/
│       └── raw_json/
└── processed/
```

The NBA collector also rebuilds `draft_all.parquet`, `base_all.parquet`, and
`advanced_all.parquet`. Those rollups include every matching period file
already in the selected output directory, not only the range from the latest
command.

## Resume and check a download

The collectors are designed for partial reruns. Without a force option they
skip existing Parquet files; the college collector additionally replaces files
that are unreadable or empty. With `--force` (Python) or a final `true` (R), the
requested files are downloaded again.

An individual endpoint can fail after its retries while the rest of the run
continues. Most such failures create a `*.failed.txt` marker, so a successful
process exit does not by itself prove that every period downloaded. Check for
markers after every run.

Windows PowerShell:

```powershell
Get-ChildItem data\raw -Recurse -Filter *.failed.txt
```

Linux or macOS:

```bash
find data/raw -name '*.failed.txt' -print
```

Rerun the same collector to retry missing files; completed Parquets are skipped.
The crosswalk collector can leave a stale marker after a later successful
download, so confirm whether its corresponding Parquet now exists.

Then verify that every downloaded Parquet can be read:

```bash
python validate_collection.py --root data/raw
```

The validator prints each file's dimensions and a final read-error count. It is
a readability check only: it does not verify expected year coverage, flag
zero-row tables, inspect raw JSON, inspect failure markers, or return a nonzero
status when a read error is reported. Review its output and the marker scan.

Before building datasets, confirm that the intended periods are complete. For
the project range, that means each NCAA output for 2003-2022, every draft and
all four Combine sources for 2009-2022, and both NBA stat measures for 2009-10
through 2024-25. Do not silently proceed with an unexplained gap: a missing
draft file can label players as undrafted, a missing Base season can reduce a
minutes outcome to zero, and a missing Combine source can shrink the four-source
classification cohort.

## Build the available processed tables

Downloading and processing are separate steps. After the raw download, build
the drafted-player minutes table and the enriched Combine classification table:

```bash
python build_model_dataset_v3.py --raw-root data/raw --processed-root data/processed --draft-start 2009 --draft-end 2022 --nba-target-years 3
python build_draft_classification_dataset.py --raw-root data/raw --processed-root data/processed --draft-start 2009 --draft-end 2022
python build_enriched_draft_classification_dataset.py --raw-root data/raw --processed-root data/processed --draft-start 2009 --draft-end 2022
```

Run the base classification builder explicitly as shown whenever raw Combine or
draft files change. The enriched builder otherwise reuses an existing base
Parquet.

These commands create:

- `drafted_players_all.parquet` and
  `model_dataset_drafted_college.parquet`;
- `draft_classification_dataset.parquet` and
  `draft_classification_enriched_dataset.parquet`; and
- audit CSVs under `data/processed/audit/`.

Before modeling, inspect
`data/processed/audit/draft_classification/combine_file_coverage.csv` and
`combine_source_key_mismatches.csv` in the same directory, along with the other
generated audits.

The frozen model configs in the root project use later, materialized advanced
snapshots named `draft_classification_advanced.parquet` and
`drafted_players_advanced.parquet`. No tracked script currently creates those
two exact files; the commands above reproduce the current base/enriched tables,
not the frozen advanced snapshots.

## Troubleshooting

- **`Rscript` is not found:** install R and add its executable directory to
  `PATH`, then open a new terminal.
- **R reports a missing package or old `hoopR`:** run the update command in
  [Prerequisites](#prerequisites), then retry.
- **NBA requests time out or are throttled:** rerun the command. Good Parquets
  are skipped. Increase `--sleep-min` and `--sleep-max` if failures persist.
- **Only Combine shooting fails:** use `--skip-combine-shooting`, then run the
  full NBA command later.
- **An NBA Parquet exists but its raw JSON is absent:** the Parquet is written
  first and a normal rerun will skip it. Rerun the relevant range with
  `--force` if both representations are required.
- **An existing NBA or crosswalk Parquet is corrupt or empty:** those collectors
  skip on file existence, so rerun the relevant range with the force option.

## Identity matching and research scope

Normalize ESPN IDs as strings before joining the college data to the NBA
crosswalk:

```python
import pandas as pd

college = pd.read_parquet(
    "data/raw/college/player_box/player_box_2022.parquet"
)
crosswalk = pd.read_parquet(
    "data/raw/crosswalks/nba_player_crosswalk.parquet"
)

college["espn_athlete_id"] = (
    college["athlete_id"].astype("Int64").astype("string")
)
crosswalk["espn_athlete_id"] = crosswalk["espn_athlete_id"].astype("string")

college_with_nba_id = college.merge(
    crosswalk[["espn_athlete_id", "nba_player_id"]].drop_duplicates(),
    on="espn_athlete_id",
    how="left",
)
```

Most college players do not have an NBA ID. That is expected: the collection
includes all available NCAA players so the project can model both drafted and
undrafted Combine participants. Ambiguous fallback matches should be reviewed
manually.

See the [root README](../README.md) for the two prediction tasks, frozen results,
and modeling limitations.
