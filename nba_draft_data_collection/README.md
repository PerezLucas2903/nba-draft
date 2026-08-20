# NBA Draft Portfolio — Data Collection

Pipeline for collecting pre-draft men's college basketball data and post-draft NBA outcomes.

## What it collects

### NCAA / men's college basketball — hoopR

For each season:

- `player_box`: game-level player box scores
- `player_stats`: cleaned season-level player stats
- `rosters`: player/roster metadata
- `team_stats`: team-level season context

The default range is **2009–2022**.

`hoopR` uses the 4-digit year associated with the season ending in that year.
For example, the NCAA season ending in spring 2022 is `season = 2022`.

### Crosswalks — hoopR

Two particularly useful identity tables are downloaded:

- `mbb_player_crosswalk.parquet`
- `nba_player_crosswalk.parquet`

The NBA crosswalk maps ESPN athlete IDs to NBA.com player IDs. Because the
college box scores also contain ESPN `athlete_id`, this can become the main
bridge between college and NBA identities instead of relying only on names.

### NBA — nba_api

For each draft year:

- official NBA draft history (`PERSON_ID`, pick, team, organization)
- combine anthropometrics
- combine strength/agility drills
- combine spot shooting
- combine non-stationary shooting

For NBA seasons:

- `LeagueDashPlayerStats` with `MeasureType=Base`
- `LeagueDashPlayerStats` with `MeasureType=Advanced`

By default, if drafts end in 2022, NBA seasons are collected through **2024-25**,
which provides three complete NBA seasons after the 2022 draft.

Both Parquet and gzipped raw JSON are retained for `nba_api`.

## Directory layout

```text
data/raw/
├── college/
│   ├── player_box/
│   ├── player_stats/
│   ├── rosters/
│   └── team_stats/
├── crosswalks/
│   ├── mbb_player_crosswalk.parquet
│   └── nba_player_crosswalk.parquet
└── nba/
    ├── draft/
    ├── combine/
    │   ├── anthro/
    │   ├── drills/
    │   ├── spot_shooting/
    │   └── nonstationary_shooting/
    ├── player_stats/
    │   ├── base/
    │   └── advanced/
    └── raw_json/
```

## Setup

Python 3.10+ is required by the current `nba_api`.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install R dependencies:

```bash
Rscript install_r_packages.R
```

## Run everything

```bash
python collect_all.py --draft-start 2009 --draft-end 2022
```

The scripts are idempotent: existing Parquet files are skipped unless `--force`
is used. A failed request gets a `.failed.txt` marker, while the remaining
historical seasons continue downloading.

## Run only NCAA data

```bash
Rscript collect_college.R 2009 2022 data/raw/college
```

Force overwrite:

```bash
Rscript collect_college.R 2009 2022 data/raw/college true
```

## Run only identity crosswalks

```bash
Rscript collect_crosswalks.R 2009 2022 2024 data/raw/crosswalks
```

## Run only NBA data

```bash
python collect_nba.py \
  --draft-start 2009 \
  --draft-end 2022 \
  --output data/raw/nba
```

If the combine shooting endpoints are unstable, start with anthropometrics
and drills:

```bash
python collect_nba.py \
  --draft-start 2009 \
  --draft-end 2022 \
  --skip-combine-shooting
```

## Validate downloaded Parquet files

```bash
python validate_collection.py --root data/raw
```

## Joining college and NBA identities

First normalize ESPN IDs to strings, then use the NBA crosswalk:

```python
import pandas as pd

college = pd.read_parquet(
    "data/raw/college/player_box/player_box_2022.parquet"
)
xwalk = pd.read_parquet(
    "data/raw/crosswalks/nba_player_crosswalk.parquet"
)

college["espn_athlete_id"] = (
    college["athlete_id"].astype("Int64").astype("string")
)
xwalk["espn_athlete_id"] = xwalk["espn_athlete_id"].astype("string")

college_with_nba_id = college.merge(
    xwalk[["espn_athlete_id", "nba_player_id"]].drop_duplicates(),
    on="espn_athlete_id",
    how="left",
)
```

Do not assume every college player has an NBA ID. Most do not. For players
without a crosswalk match, use draft history plus name/team/year as a
secondary matching route and manually review ambiguous cases.

## Why all NCAA players are collected

This downloads **all available NCAA players**, not only drafted players.
That is deliberate.

It lets you later choose between:

1. performance among drafted players; or
2. a full prospect pipeline, including undrafted players who did or did not reach the NBA.

## How this supports the two-stage research design

The modeling repository uses these sources to separate two questions:

1. **Entry:** NCAA and Combine information predicts whether a Combine
   participant is drafted in the same year.
2. **Opportunity:** among drafted participants, the same NCAA feature contract
   predicts NBA minutes over the next three seasons.

The harmonized comparison table is generated at
`data/processed/two_stage_college_comparison.parquet`. Draft position is not a
college predictor and is excluded from the primary comparison; it is introduced
only in a separately labeled sensitivity model. See the root
root `README.md` and `notebooks/two_stage_analysis.ipynb` for the findings and
limitations.
