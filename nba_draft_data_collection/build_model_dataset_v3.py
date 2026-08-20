from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def normalize_id(s: pd.Series) -> pd.Series:
    """Normalize numeric/string identifiers to nullable strings without '.0'."""
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    x = unicodedata.normalize("NFKD", str(value))
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.lower()
    x = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.astype(float)
    num = num.astype(float)
    return num.div(den.where(den != 0))


def read_many(pattern: str | Path) -> pd.DataFrame:
    files = sorted(Path().glob(str(pattern))) if not isinstance(pattern, Path) else []
    if isinstance(pattern, Path):
        files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def read_many_dir(directory: Path, glob_pattern: str) -> pd.DataFrame:
    files = sorted(directory.glob(glob_pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def first_existing(df: pd.DataFrame, *names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def parse_year_from_filename(path: Path) -> int:
    m = re.search(r"(\d{4})", path.stem)
    if not m:
        raise ValueError(f"Could not extract year from {path.name}")
    return int(m.group(1))


# ---------------------------------------------------------------------
# College features
# ---------------------------------------------------------------------

COLLEGE_SUM_COLS = [
    "games_on_roster",
    "games_played",
    "games_started",
    "minutes",
    "field_goals_made",
    "field_goals_attempted",
    "three_point_field_goals_made",
    "three_point_field_goals_attempted",
    "free_throws_made",
    "free_throws_attempted",
    "offensive_rebounds",
    "defensive_rebounds",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    "points",
]


def add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["fg_pct"] = safe_div(df["field_goals_made"], df["field_goals_attempted"])
    df["three_pct"] = safe_div(
        df["three_point_field_goals_made"],
        df["three_point_field_goals_attempted"],
    )
    df["ft_pct"] = safe_div(df["free_throws_made"], df["free_throws_attempted"])

    df["efg_pct"] = safe_div(
        df["field_goals_made"] + 0.5 * df["three_point_field_goals_made"],
        df["field_goals_attempted"],
    )

    ts_den = 2 * (
        df["field_goals_attempted"] + 0.44 * df["free_throws_attempted"]
    )
    df["ts_pct"] = safe_div(df["points"], ts_den)

    for source, name in [
        ("points", "points_per_40"),
        ("rebounds", "rebounds_per_40"),
        ("assists", "assists_per_40"),
        ("steals", "steals_per_40"),
        ("blocks", "blocks_per_40"),
        ("turnovers", "turnovers_per_40"),
    ]:
        df[name] = safe_div(40 * df[source], df["minutes"])

    df["assist_turnover_ratio"] = safe_div(df["assists"], df["turnovers"])
    df["three_attempt_rate"] = safe_div(
        df["three_point_field_goals_attempted"],
        df["field_goals_attempted"],
    )
    df["free_throw_rate"] = safe_div(
        df["free_throws_attempted"],
        df["field_goals_attempted"],
    )
    df["minutes_per_game"] = safe_div(df["minutes"], df["games_played"])

    return df


def load_college_seasons(root: Path) -> pd.DataFrame:
    directory = root / "college" / "player_season_from_box"
    df = read_many_dir(directory, "player_season_from_box_*.parquet")
    if df.empty:
        raise FileNotFoundError(
            f"No derived college season files found in {directory}. "
            "Run the fixed collect_college.R first."
        )

    df.columns = [str(c) for c in df.columns]
    df["athlete_id"] = normalize_id(df["athlete_id"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["team_id"] = normalize_id(df["team_id"])

    # player_season_from_box has one row per player-team-season. A transfer can
    # therefore appear multiple times in one season. Collapse to player-season
    # by summing raw counts and recomputing all rates.
    for col in COLLEGE_SUM_COLS:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    name_col = first_existing(df, "athlete_display_name", "player_name")
    pos_col = first_existing(
        df,
        "athlete_position_abbreviation",
        "athlete_position_name",
    )

    agg = {c: "sum" for c in COLLEGE_SUM_COLS}
    if name_col:
        agg[name_col] = "first"
    if pos_col:
        agg[pos_col] = "first"

    collapsed = (
        df.groupby(["athlete_id", "season"], dropna=False, as_index=False)
        .agg(agg)
    )

    if name_col and name_col != "athlete_display_name":
        collapsed = collapsed.rename(columns={name_col: "athlete_display_name"})
    if pos_col and pos_col != "athlete_position_abbreviation":
        collapsed = collapsed.rename(
            columns={pos_col: "athlete_position_abbreviation"}
        )

    collapsed = add_rate_features(collapsed)
    collapsed["normalized_name"] = collapsed["athlete_display_name"].map(
        normalize_name
    )
    return collapsed



def load_college_team_rows(root: Path) -> pd.DataFrame:
    """
    Load the derived player-team-season rows before they are collapsed to
    player-season. These are used only for identity disambiguation.
    """
    directory = root / "college" / "player_season_from_box"
    df = read_many_dir(directory, "player_season_from_box_*.parquet")
    if df.empty:
        return df

    df["athlete_id"] = normalize_id(df["athlete_id"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    keep = [
        c
        for c in [
            "athlete_id",
            "season",
            "athlete_display_name",
            "team_id",
            "team_display_name",
        ]
        if c in df.columns
    ]
    return df[keep].drop_duplicates()

def load_player_core(root: Path) -> pd.DataFrame:
    directory = root / "college" / "player_core"
    df = read_many_dir(directory, "player_core_*.parquet")
    if df.empty:
        return df

    df["athlete_id"] = normalize_id(df["athlete_id"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    keep = [
        c
        for c in [
            "athlete_id",
            "season",
            "date_of_birth",
            "position_abbreviation",
            "position_name",
            "full_name",
        ]
        if c in df.columns
    ]
    return df[keep].drop_duplicates(["athlete_id", "season"], keep="last")


# ---------------------------------------------------------------------
# Draft + identity mapping
# ---------------------------------------------------------------------

def load_draft(root: Path, start_year: int, end_year: int) -> pd.DataFrame:
    path = root / "nba" / "draft" / "draft_all.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path).copy()

    required = ["PERSON_ID", "PLAYER_NAME", "SEASON", "OVERALL_PICK"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Draft file is missing columns: {missing}")

    df["nba_player_id"] = normalize_id(df["PERSON_ID"])
    df["draft_year"] = pd.to_numeric(df["SEASON"], errors="coerce").astype("Int64")
    df["overall_pick"] = pd.to_numeric(df["OVERALL_PICK"], errors="coerce")
    df["round_number"] = pd.to_numeric(df.get("ROUND_NUMBER"), errors="coerce")
    df["round_pick"] = pd.to_numeric(df.get("ROUND_PICK"), errors="coerce")
    df["normalized_name"] = df["PLAYER_NAME"].map(normalize_name)

    df = df[df["draft_year"].between(start_year, end_year)].copy()

    keep = [
        "nba_player_id",
        "PLAYER_NAME",
        "normalized_name",
        "draft_year",
        "overall_pick",
        "round_number",
        "round_pick",
        "TEAM_ABBREVIATION",
        "ORGANIZATION",
        "ORGANIZATION_TYPE",
    ]
    keep = [c for c in keep if c in df.columns]

    return df[keep].drop_duplicates("nba_player_id")


def load_nba_crosswalk(root: Path, audit_dir: Path) -> pd.DataFrame:
    path = root / "crosswalks" / "nba_player_crosswalk.parquet"
    empty = pd.DataFrame(columns=["nba_player_id", "college_espn_id"])

    if not path.exists():
        print("WARNING: NBA player crosswalk not found; using conservative name/team fallback.")
        return empty

    x = pd.read_parquet(path).copy()

    # hoopR can return/save a zero-row, zero-column result when the cached
    # crosswalk artifacts are unavailable. Treat that as 'crosswalk unavailable'
    # instead of aborting the whole dataset build.
    if x.empty or len(x.columns) == 0:
        print(
            "WARNING: NBA player crosswalk is empty; "
            "using conservative name/team fallback."
        )
        return empty

    if "nba_player_id" not in x or "espn_athlete_id" not in x:
        print(
            "WARNING: NBA player crosswalk has unexpected columns; "
            "using conservative name/team fallback."
        )
        x.to_csv(audit_dir / "unexpected_nba_crosswalk.csv", index=False)
        return empty

    x["nba_player_id"] = normalize_id(x["nba_player_id"])
    x["espn_athlete_id"] = normalize_id(x["espn_athlete_id"])
    x = x.dropna(subset=["nba_player_id", "espn_athlete_id"])

    # Choose the most frequent mapping across seasons, but save ambiguous mappings.
    counts = (
        x.groupby(["nba_player_id", "espn_athlete_id"])
        .size()
        .rename("n")
        .reset_index()
    )
    n_ids = counts.groupby("nba_player_id")["espn_athlete_id"].nunique()
    ambiguous_ids = set(n_ids[n_ids > 1].index)

    if ambiguous_ids:
        audit_dir.mkdir(parents=True, exist_ok=True)
        counts[counts["nba_player_id"].isin(ambiguous_ids)].to_csv(
            audit_dir / "ambiguous_nba_crosswalk.csv", index=False
        )

    best = (
        counts.sort_values(
            ["nba_player_id", "n", "espn_athlete_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("nba_player_id")
        .rename(columns={"espn_athlete_id": "college_espn_id"})
    )

    return best[["nba_player_id", "college_espn_id"]]


def add_conservative_name_fallback(
    draft: pd.DataFrame,
    college: pd.DataFrame,
    college_team_rows: pd.DataFrame,
    audit_dir: Path,
) -> pd.DataFrame:
    """
    Match draft players not found in the NBA crosswalk.

    Rules:
      1. exact normalized player-name match only;
      2. candidate NCAA seasons must be in [draft_year - 6, draft_year];
      3. if exactly one ESPN athlete ID exists -> accept;
      4. if several IDs exist, use draft ORGANIZATION vs NCAA team name as
         a disambiguator;
      5. otherwise leave unmatched and write candidates to an audit CSV.

    No fuzzy player-name matching is performed automatically.
    """
    out = draft.copy()

    if "college_espn_id" not in out:
        out["college_espn_id"] = pd.NA

    out["identity_match_method"] = np.where(
        out["college_espn_id"].notna(),
        "nba_crosswalk",
        pd.NA,
    )

    # Candidate lookup by exact normalized name.
    by_name = {
        name: g
        for name, g in college[college["normalized_name"] != ""].groupby(
            "normalized_name"
        )
    }

    # Team rows retain team_display_name, which the player-season aggregate
    # intentionally drops.
    team_lookup = college_team_rows.copy()
    if not team_lookup.empty:
        team_lookup["athlete_id"] = normalize_id(team_lookup["athlete_id"])
        team_lookup["season"] = pd.to_numeric(
            team_lookup["season"], errors="coerce"
        ).astype("Int64")
        if "team_display_name" in team_lookup:
            team_lookup["normalized_team"] = team_lookup[
                "team_display_name"
            ].map(normalize_name)
        else:
            team_lookup["normalized_team"] = ""

    ambiguous_rows = []

    for idx in out.index[out["college_espn_id"].isna()]:
        row = out.loc[idx]
        candidates = by_name.get(row["normalized_name"])

        if candidates is None:
            continue

        draft_year = int(row["draft_year"])
        candidates = candidates[
            candidates["season"].between(draft_year - 6, draft_year)
        ]

        ids = list(candidates["athlete_id"].dropna().unique())

        if len(ids) == 1:
            out.at[idx, "college_espn_id"] = ids[0]
            out.at[idx, "identity_match_method"] = "exact_name_unique"
            continue

        # If multiple ESPN IDs share the exact same normalized name, try
        # matching the college listed in NBA DraftHistory ORGANIZATION.
        org = normalize_name(row.get("ORGANIZATION", ""))

        if len(ids) > 1 and org and not team_lookup.empty:
            team_candidates = team_lookup[
                team_lookup["athlete_id"].isin(ids)
                & team_lookup["season"].between(draft_year - 6, draft_year)
            ].copy()

            # Require a reasonably strong deterministic string relationship.
            # Examples: "Duke" vs "Duke Blue Devils", "Kentucky" vs
            # "Kentucky Wildcats".
            team_candidates["org_match"] = team_candidates[
                "normalized_team"
            ].map(
                lambda team: bool(
                    team
                    and (
                        org == team
                        or org in team
                        or team in org
                    )
                )
            )

            matched_ids = list(
                team_candidates.loc[
                    team_candidates["org_match"], "athlete_id"
                ].dropna().unique()
            )

            if len(matched_ids) == 1:
                out.at[idx, "college_espn_id"] = matched_ids[0]
                out.at[idx, "identity_match_method"] = "exact_name_plus_org"
                continue

        # Keep the case for manual review rather than guessing.
        for athlete_id in ids:
            cand = candidates[candidates["athlete_id"] == athlete_id]
            ambiguous_rows.append(
                {
                    "nba_player_id": row["nba_player_id"],
                    "draft_player_name": row["PLAYER_NAME"],
                    "draft_year": row["draft_year"],
                    "draft_organization": row.get("ORGANIZATION", pd.NA),
                    "candidate_espn_athlete_id": athlete_id,
                    "candidate_name": (
                        cand["athlete_display_name"].iloc[-1]
                        if "athlete_display_name" in cand
                        else pd.NA
                    ),
                    "candidate_first_season": cand["season"].min(),
                    "candidate_last_season": cand["season"].max(),
                }
            )

    if ambiguous_rows:
        audit_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ambiguous_rows).to_csv(
            audit_dir / "ambiguous_name_matches.csv",
            index=False,
        )

    return out


# ---------------------------------------------------------------------
# College feature table relative to each player's draft year
# ---------------------------------------------------------------------

def build_college_features(
    draft: pd.DataFrame,
    college: pd.DataFrame,
    player_core: pd.DataFrame,
) -> pd.DataFrame:
    mapped = draft.dropna(subset=["college_espn_id"]).copy()

    hist = mapped[
        ["nba_player_id", "college_espn_id", "draft_year"]
    ].merge(
        college,
        left_on="college_espn_id",
        right_on="athlete_id",
        how="left",
    )

    hist = hist[
        hist["season"].notna()
        & (hist["season"] <= hist["draft_year"])
        & (hist["season"] >= hist["draft_year"] - 6)
    ].copy()

    if hist.empty:
        return pd.DataFrame(columns=["nba_player_id"])

    # Last college season features.
    last = (
        hist.sort_values(["nba_player_id", "season"])
        .groupby("nba_player_id", as_index=False)
        .tail(1)
        .copy()
    )

    last_feature_cols = [
        c
        for c in [
            "season",
            "games_played",
            "games_started",
            "minutes",
            "minutes_per_game",
            "points",
            "rebounds",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "fg_pct",
            "three_pct",
            "ft_pct",
            "efg_pct",
            "ts_pct",
            "points_per_40",
            "rebounds_per_40",
            "assists_per_40",
            "steals_per_40",
            "blocks_per_40",
            "turnovers_per_40",
            "assist_turnover_ratio",
            "three_attempt_rate",
            "free_throw_rate",
            "athlete_position_abbreviation",
        ]
        if c in last.columns
    ]

    last_out = last[["nba_player_id", "college_espn_id"] + last_feature_cols].copy()
    last_out = last_out.rename(
        columns={
            c: (
                "college_last_season"
                if c == "season"
                else f"college_last_{c}"
            )
            for c in last_feature_cols
        }
    )

    # Career totals through the draft.
    for col in COLLEGE_SUM_COLS:
        if col not in hist:
            hist[col] = 0.0

    career = (
        hist.groupby("nba_player_id", as_index=False)
        .agg(
            college_career_seasons=("season", "nunique"),
            college_career_first_season=("season", "min"),
            college_career_last_season=("season", "max"),
            **{
                f"college_career_{c}": (c, "sum")
                for c in COLLEGE_SUM_COLS
            },
        )
    )

    # Recompute career rates from totals.
    career["college_career_fg_pct"] = safe_div(
        career["college_career_field_goals_made"],
        career["college_career_field_goals_attempted"],
    )
    career["college_career_three_pct"] = safe_div(
        career["college_career_three_point_field_goals_made"],
        career["college_career_three_point_field_goals_attempted"],
    )
    career["college_career_ft_pct"] = safe_div(
        career["college_career_free_throws_made"],
        career["college_career_free_throws_attempted"],
    )
    career["college_career_efg_pct"] = safe_div(
        career["college_career_field_goals_made"]
        + 0.5 * career["college_career_three_point_field_goals_made"],
        career["college_career_field_goals_attempted"],
    )

    ts_den = 2 * (
        career["college_career_field_goals_attempted"]
        + 0.44 * career["college_career_free_throws_attempted"]
    )
    career["college_career_ts_pct"] = safe_div(
        career["college_career_points"], ts_den
    )

    for source, name in [
        ("points", "points_per_40"),
        ("rebounds", "rebounds_per_40"),
        ("assists", "assists_per_40"),
        ("steals", "steals_per_40"),
        ("blocks", "blocks_per_40"),
        ("turnovers", "turnovers_per_40"),
    ]:
        career[f"college_career_{name}"] = safe_div(
            40 * career[f"college_career_{source}"],
            career["college_career_minutes"],
        )

    career["college_career_assist_turnover_ratio"] = safe_div(
        career["college_career_assists"],
        career["college_career_turnovers"],
    )
    career["college_career_three_attempt_rate"] = safe_div(
        career["college_career_three_point_field_goals_attempted"],
        career["college_career_field_goals_attempted"],
    )
    career["college_career_free_throw_rate"] = safe_div(
        career["college_career_free_throws_attempted"],
        career["college_career_field_goals_attempted"],
    )

    result = last_out.merge(career, on="nba_player_id", how="left")

    # Stable bio features from player_core. Avoid current height/weight here;
    # combine measurements are preferable pre-draft physical features.
    if not player_core.empty:
        core = player_core.copy()
        core["athlete_id"] = normalize_id(core["athlete_id"])

        bio = result[
            ["nba_player_id", "college_espn_id", "college_last_season"]
        ].merge(
            core,
            left_on=["college_espn_id", "college_last_season"],
            right_on=["athlete_id", "season"],
            how="left",
        )

        bio_keep = ["nba_player_id"]
        if "date_of_birth" in bio:
            # ESPN/hoopR birth dates may arrive timezone-aware (e.g. UTC),
            # while dates constructed locally are timezone-naive. Normalize
            # both to timezone-naive timestamps before subtraction.
            bio["date_of_birth"] = (
                pd.to_datetime(
                    bio["date_of_birth"],
                    errors="coerce",
                    utc=True,
                )
                .dt.tz_convert(None)
            )

            draft_year_map = draft.set_index("nba_player_id")["draft_year"]
            bio["draft_year"] = bio["nba_player_id"].map(draft_year_map)

            reference_date = pd.to_datetime(
                bio["draft_year"].astype("Int64").astype("string") + "-07-01",
                errors="coerce",
            )

            bio["age_on_draft_year_july1"] = (
                (reference_date - bio["date_of_birth"]).dt.days / 365.2425
            )
            bio_keep.append("age_on_draft_year_july1")

        for c in ["position_abbreviation", "position_name"]:
            if c in bio.columns:
                new_c = f"college_core_{c}"
                bio[new_c] = bio[c]
                bio_keep.append(new_c)

        bio = bio[bio_keep].drop_duplicates("nba_player_id")
        result = result.merge(bio, on="nba_player_id", how="left")

    return result


# ---------------------------------------------------------------------
# Combine features
# ---------------------------------------------------------------------

def load_combine_source(directory: Path, prefix: str) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    files = [p for p in files if "_all" not in p.stem]
    if not files:
        return pd.DataFrame(columns=["nba_player_id", "draft_year"])

    chunks = []
    for p in files:
        df = pd.read_parquet(p).copy()
        if df.empty or "PLAYER_ID" not in df.columns:
            continue

        df["nba_player_id"] = normalize_id(df["PLAYER_ID"])
        df["draft_year"] = parse_year_from_filename(p)

        ignore = {
            "TEMP_PLAYER_ID",
            "PLAYER_ID",
            "FIRST_NAME",
            "LAST_NAME",
            "PLAYER_NAME",
            "POSITION",
        }

        numeric_cols = []
        for c in df.columns:
            if c in ignore or c in {"nba_player_id", "draft_year"}:
                continue
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().any():
                df[c] = converted
                numeric_cols.append(c)

        if not numeric_cols:
            continue

        # Usually one row per player. If a source ever contains duplicates,
        # numeric mean is a conservative way to collapse them.
        reduced = (
            df.groupby(["nba_player_id", "draft_year"], as_index=False)[
                numeric_cols
            ]
            .mean()
        )
        reduced = reduced.rename(
            columns={c: f"combine_{prefix}_{c.lower()}" for c in numeric_cols}
        )
        chunks.append(reduced)

    if not chunks:
        return pd.DataFrame(columns=["nba_player_id", "draft_year"])

    out = pd.concat(chunks, ignore_index=True)
    # Same year should only occur once because each year has one file.
    return out.drop_duplicates(["nba_player_id", "draft_year"])


def build_combine_features(root: Path) -> pd.DataFrame:
    combine_root = root / "nba" / "combine"
    sources = [
        ("anthro", "anthro"),
        ("drills", "drill"),
        ("spot_shooting", "spot"),
        ("nonstationary_shooting", "nonstationary"),
    ]

    result = None
    for folder, prefix in sources:
        df = load_combine_source(combine_root / folder, prefix)
        if df.empty:
            continue
        if result is None:
            result = df
        else:
            result = result.merge(
                df,
                on=["nba_player_id", "draft_year"],
                how="outer",
            )

    if result is None:
        return pd.DataFrame(columns=["nba_player_id", "draft_year"])

    result["combine_available"] = 1
    return result


# ---------------------------------------------------------------------
# NBA post-draft targets
# ---------------------------------------------------------------------

NBA_BASE_TOTAL_COLS = [
    "GP",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
]


def load_nba_base(root: Path) -> pd.DataFrame:
    path = root / "nba" / "player_stats" / "base" / "base_all.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path).copy()
    df["nba_player_id"] = normalize_id(df["PLAYER_ID"])
    df["nba_season_start"] = pd.to_numeric(
        df["SEASON"].astype("string").str.slice(0, 4),
        errors="coerce",
    ).astype("Int64")

    for c in NBA_BASE_TOTAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def build_nba_targets(
    draft: pd.DataFrame,
    nba_base: pd.DataFrame,
    years: int = 3,
) -> pd.DataFrame:
    tmp = draft[["nba_player_id", "draft_year"]].merge(
        nba_base,
        on="nba_player_id",
        how="left",
    )

    in_window = (
        tmp["nba_season_start"].notna()
        & (tmp["nba_season_start"] >= tmp["draft_year"])
        & (tmp["nba_season_start"] < tmp["draft_year"] + years)
    )
    tmp = tmp[in_window].copy()

    sum_cols = [c for c in NBA_BASE_TOTAL_COLS if c in tmp.columns]
    agg_spec = {f"nba_{c.lower()}_{years}y": (c, "sum") for c in sum_cols}
    agg_spec[f"nba_seasons_played_{years}y"] = ("nba_season_start", "nunique")

    if tmp.empty:
        out = draft[["nba_player_id"]].copy()
    else:
        out = tmp.groupby("nba_player_id", as_index=False).agg(**agg_spec)

    # Keep every drafted player. Players with no NBA row in the window get 0.
    out = draft[["nba_player_id"]].merge(out, on="nba_player_id", how="left")

    zero_cols = [
        c
        for c in out.columns
        if c.startswith("nba_") and c != "nba_player_id"
    ]
    out[zero_cols] = out[zero_cols].fillna(0.0)

    gp = f"nba_gp_{years}y"
    mins = f"nba_min_{years}y"
    pts = f"nba_pts_{years}y"
    reb = f"nba_reb_{years}y"
    ast = f"nba_ast_{years}y"

    if gp in out:
        out[f"nba_reached_league_{years}y"] = (out[gp] > 0).astype(int)
    if mins in out and gp in out:
        out[f"nba_minutes_per_game_{years}y"] = safe_div(out[mins], out[gp])
    if mins in out and pts in out:
        out[f"nba_points_per_36_{years}y"] = safe_div(36 * out[pts], out[mins])
    if mins in out and reb in out:
        out[f"nba_rebounds_per_36_{years}y"] = safe_div(36 * out[reb], out[mins])
    if mins in out and ast in out:
        out[f"nba_assists_per_36_{years}y"] = safe_div(36 * out[ast], out[mins])

    return out


def load_nba_advanced(root: Path) -> pd.DataFrame:
    path = root / "nba" / "player_stats" / "advanced" / "advanced_all.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path).copy()
    df["nba_player_id"] = normalize_id(df["PLAYER_ID"])
    df["nba_season_start"] = pd.to_numeric(
        df["SEASON"].astype("string").str.slice(0, 4),
        errors="coerce",
    ).astype("Int64")

    for c in df.columns:
        if c not in {"PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
                     "AGE", "SEASON", "nba_player_id", "nba_season_start"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group[weight_col], errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def build_nba_advanced_targets(
    draft: pd.DataFrame,
    advanced: pd.DataFrame,
    years: int = 3,
) -> pd.DataFrame:
    if advanced.empty:
        return draft[["nba_player_id"]].copy()

    tmp = draft[["nba_player_id", "draft_year"]].merge(
        advanced,
        on="nba_player_id",
        how="left",
    )
    tmp = tmp[
        tmp["nba_season_start"].notna()
        & (tmp["nba_season_start"] >= tmp["draft_year"])
        & (tmp["nba_season_start"] < tmp["draft_year"] + years)
    ].copy()

    candidates = [
        c
        for c in [
            "TS_PCT",
            "EFG_PCT",
            "USG_PCT",
            "AST_PCT",
            "REB_PCT",
            "OREB_PCT",
            "DREB_PCT",
            "OFF_RATING",
            "DEF_RATING",
            "NET_RATING",
            "PACE",
            "PIE",
        ]
        if c in tmp.columns
    ]
    weight_col = "MIN" if "MIN" in tmp.columns else None

    if tmp.empty or not candidates or weight_col is None:
        return draft[["nba_player_id"]].copy()

    rows = []
    for player_id, g in tmp.groupby("nba_player_id"):
        row = {"nba_player_id": player_id}
        for c in candidates:
            row[f"nba_{c.lower()}_{years}y_weighted"] = weighted_mean(
                g, c, weight_col
            )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Final dataset
# ---------------------------------------------------------------------

def build_dataset(
    raw_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
    nba_target_years: int,
) -> pd.DataFrame:
    processed_root.mkdir(parents=True, exist_ok=True)
    audit_dir = processed_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("Loading draft...")
    draft = load_draft(raw_root, start_year, end_year)

    print("Loading college seasons...")
    college = load_college_seasons(raw_root)
    college_team_rows = load_college_team_rows(raw_root)
    core = load_player_core(raw_root)

    print("Matching NBA IDs to college ESPN IDs...")
    xwalk = load_nba_crosswalk(raw_root, audit_dir)
    draft = draft.merge(xwalk, on="nba_player_id", how="left")
    draft = add_conservative_name_fallback(
        draft,
        college,
        college_team_rows,
        audit_dir,
    )

    print("Building college features...")
    college_features = build_college_features(draft, college, core)

    print("Building combine features...")
    combine = build_combine_features(raw_root)

    print("Building NBA targets...")
    nba_base = load_nba_base(raw_root)
    nba_targets = build_nba_targets(
        draft,
        nba_base,
        years=nba_target_years,
    )
    nba_advanced = load_nba_advanced(raw_root)
    nba_adv_targets = build_nba_advanced_targets(
        draft,
        nba_advanced,
        years=nba_target_years,
    )

    final = (
        draft
        .merge(college_features, on=["nba_player_id", "college_espn_id"], how="left")
        .merge(combine, on=["nba_player_id", "draft_year"], how="left")
        .merge(nba_targets, on="nba_player_id", how="left")
        .merge(nba_adv_targets, on="nba_player_id", how="left")
    )

    final["has_college_data"] = final["college_last_season"].notna().astype(int)
    if "combine_available" in final:
        final["combine_available"] = final["combine_available"].fillna(0).astype(int)
    else:
        final["combine_available"] = 0

    # College-only modeling population.
    model_df = final[final["has_college_data"] == 1].copy()

    # Audit files are very important: never silently discard unmatched prospects.
    final[final["has_college_data"] == 0].to_csv(
        audit_dir / "drafted_players_without_college_match.csv",
        index=False,
    )
    final[
        final["identity_match_method"].isna()
    ].to_csv(
        audit_dir / "drafted_players_without_identity_match.csv",
        index=False,
    )

    full_path = processed_root / "drafted_players_all.parquet"
    model_path = processed_root / "model_dataset_drafted_college.parquet"

    final.to_parquet(full_path, index=False)
    model_df.to_parquet(model_path, index=False)

    print()
    print(f"Drafted players:                  {len(final):,}")
    print(f"With college data:                {len(model_df):,}")
    print(
        "College match rate:               "
        f"{len(model_df) / max(len(final), 1):.1%}"
    )
    print(
        "Matched by NBA crosswalk:          "
        f"{(final['identity_match_method'] == 'nba_crosswalk').sum():,}"
    )
    print(
        "Matched by exact-name fallback:    "
        f"{(final['identity_match_method'] == 'exact_name_unique').sum():,}"
    )
    print(
        "Matched by name + organization:    "
        f"{(final['identity_match_method'] == 'exact_name_plus_org').sum():,}"
    )
    print(f"Saved: {full_path}")
    print(f"Saved: {model_path}")
    print(f"Audit: {audit_dir}")

    return model_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    p.add_argument("--draft-start", type=int, default=2009)
    p.add_argument("--draft-end", type=int, default=2022)
    p.add_argument("--nba-target-years", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        start_year=args.draft_start,
        end_year=args.draft_end,
        nba_target_years=args.nba_target_years,
    )


if __name__ == "__main__":
    main()
