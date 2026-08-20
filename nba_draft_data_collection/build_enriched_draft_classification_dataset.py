"""Build a target-independent combine + college draft classifier dataset.

The base population and binary target come from
``draft_classification_dataset.parquet``.  College identity is resolved without
using that target, Draft History metadata, an outcome-conditioned crosswalk, or
fuzzy matching:

1. exact compact name in the draft-year NCAA season, when it identifies one
   ESPN athlete;
2. the parenthetical ``UNC``/``UMD`` qualifiers present in two combine names,
   but only when the qualifier identifies one same-season athlete by team; or
3. exact compact name in the preceding six NCAA seasons, when one athlete ID is
   unique across the whole window.

Names, ESPN athlete IDs, team IDs/names, match methods, absolute college season,
and birth dates are written to audit files only.  The model table retains the
original event key and target, combine features, target-safe player history,
age on July 1 of the draft year, college position, and primary-team context.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

try:  # Support both ``python -m ...`` and direct script execution.
    from .build_draft_classification_dataset import (
        KEY_COLUMNS,
        build_draft_classification_dataset,
        load_draft_keys,
    )
    from .build_model_dataset_v3 import normalize_id, parse_year_from_filename
except ImportError:  # pragma: no cover - used by direct script execution.
    from build_draft_classification_dataset import (
        KEY_COLUMNS,
        build_draft_classification_dataset,
        load_draft_keys,
    )
    from build_model_dataset_v3 import normalize_id, parse_year_from_filename


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = SCRIPT_DIR / "data" / "raw"
DEFAULT_PROCESSED_ROOT = SCRIPT_DIR / "data" / "processed"

PLAYER_TOTAL_COLUMNS = [
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

PLAYER_RATE_COLUMNS = [
    "minutes_per_game",
    "start_rate",
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
    "field_goal_attempts_per_40",
    "three_point_attempts_per_40",
    "free_throw_attempts_per_40",
    "offensive_rebounds_per_40",
    "defensive_rebounds_per_40",
    "fouls_per_40",
    "assist_turnover_ratio",
    "three_attempt_rate",
    "free_throw_rate",
]

DEVELOPMENT_RATE_COLUMNS = [
    "minutes_per_game",
    "start_rate",
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
]

# Only these two qualifiers occur in the combine cohort.  Keeping this map
# explicit prevents a general organization-name matcher from quietly becoming
# fuzzy identity resolution.
TEAM_QUALIFIER_ALIASES = {
    "unc": ("northcarolina",),
    "umd": ("maryland",),
}

TEAM_STAT_COLUMNS = {
    "gamesPlayed": "games_played",
    "avgPoints": "avg_points",
    "avgRebounds": "avg_rebounds",
    "avgAssists": "avg_assists",
    "avgTurnovers": "avg_turnovers",
    "avgSteals": "avg_steals",
    "avgBlocks": "avg_blocks",
    "avgFieldGoalsMade": "avg_field_goals_made",
    "avgFieldGoalsAttempted": "avg_field_goals_attempted",
    "avgThreePointFieldGoalsMade": "avg_three_point_field_goals_made",
    "avgThreePointFieldGoalsAttempted": "avg_three_point_field_goals_attempted",
    "avgFreeThrowsMade": "avg_free_throws_made",
    "avgFreeThrowsAttempted": "avg_free_throws_attempted",
    "avgOffensiveRebounds": "avg_offensive_rebounds",
    "avgDefensiveRebounds": "avg_defensive_rebounds",
    "assistTurnoverRatio": "assist_turnover_ratio",
}

TEAM_PERCENTILE_FEATURES = [
    "avg_points",
    "avg_rebounds",
    "avg_assists",
    "avg_steals",
    "avg_blocks",
    "estimated_possessions",
    "offensive_rating_est",
    "effective_fg_pct",
    "true_shooting_pct",
    "three_attempt_rate",
    "free_throw_rate",
    "turnover_rate",
    "assist_per_field_goal",
]


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    columns_if_empty: list[str] | None = None,
) -> None:
    """Write an audit CSV while retaining useful headers for empty audits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and len(frame.columns) == 0 and columns_if_empty is not None:
        frame = pd.DataFrame(columns=columns_if_empty)
    frame.to_csv(path, index=False)


def compact_name(value: object) -> str:
    """Return an accent-, punctuation-, whitespace-, and suffix-free name."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"\b(?:jr|sr|ii|iii|iv|v)\b\.?", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_combine_name(value: object) -> tuple[object, str, str]:
    """Split an optional trailing school qualifier from a combine name."""
    if pd.isna(value) or not str(value).strip():
        return pd.NA, "", ""
    raw = str(value).strip()
    match = re.search(r"\s*\(([^()]*)\)\s*$", raw)
    qualifier = match.group(1).strip() if match else ""
    base = raw[: match.start()].strip() if match else raw
    return base, compact_name(base), compact_name(qualifier)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce").astype(float)
    denominator = pd.to_numeric(denominator, errors="coerce").astype(float)
    result = numerator.div(denominator.where(denominator.ne(0)))
    return result.replace([np.inf, -np.inf], np.nan)


def add_player_rate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute rates from totals for either a season or a career."""
    result = frame.copy()
    for column in PLAYER_TOTAL_COLUMNS:
        if column not in result:
            result[column] = np.nan

    result["minutes_per_game"] = _safe_divide(
        result["minutes"], result["games_played"]
    )
    result["start_rate"] = _safe_divide(
        result["games_started"], result["games_played"]
    )
    result["fg_pct"] = _safe_divide(
        result["field_goals_made"], result["field_goals_attempted"]
    )
    result["three_pct"] = _safe_divide(
        result["three_point_field_goals_made"],
        result["three_point_field_goals_attempted"],
    )
    result["ft_pct"] = _safe_divide(
        result["free_throws_made"], result["free_throws_attempted"]
    )
    result["efg_pct"] = _safe_divide(
        result["field_goals_made"]
        + 0.5 * result["three_point_field_goals_made"],
        result["field_goals_attempted"],
    )
    result["ts_pct"] = _safe_divide(
        result["points"],
        2
        * (
            result["field_goals_attempted"]
            + 0.44 * result["free_throws_attempted"]
        ),
    )

    per_40_sources = {
        "points": "points_per_40",
        "rebounds": "rebounds_per_40",
        "assists": "assists_per_40",
        "steals": "steals_per_40",
        "blocks": "blocks_per_40",
        "turnovers": "turnovers_per_40",
        "field_goals_attempted": "field_goal_attempts_per_40",
        "three_point_field_goals_attempted": "three_point_attempts_per_40",
        "free_throws_attempted": "free_throw_attempts_per_40",
        "offensive_rebounds": "offensive_rebounds_per_40",
        "defensive_rebounds": "defensive_rebounds_per_40",
        "fouls": "fouls_per_40",
    }
    for source, feature in per_40_sources.items():
        result[feature] = _safe_divide(
            40.0 * result[source], result["minutes"]
        )

    result["assist_turnover_ratio"] = _safe_divide(
        result["assists"], result["turnovers"]
    )
    result["three_attempt_rate"] = _safe_divide(
        result["three_point_field_goals_attempted"],
        result["field_goals_attempted"],
    )
    result["free_throw_rate"] = _safe_divide(
        result["free_throws_attempted"], result["field_goals_attempted"]
    )
    return result


def _normalize_college_position(value: object) -> object:
    if pd.isna(value) or not str(value).strip():
        return pd.NA
    text = str(value).strip().upper().replace("/", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    return {"GUARD": "G", "FORWARD": "F", "CENTER": "C"}.get(text, text)


def add_combine_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add row-local physical composites and compact shooting protocols."""
    result = frame.copy()

    physical_formulas = {
        "combine_derived_wingspan_minus_height": (
            "combine_anthro_wingspan",
            "combine_anthro_height_wo_shoes",
            "subtract",
        ),
        "combine_derived_standing_reach_minus_height": (
            "combine_anthro_standing_reach",
            "combine_anthro_height_wo_shoes",
            "subtract",
        ),
        "combine_derived_max_reach": (
            "combine_anthro_standing_reach",
            "combine_drill_max_vertical_leap",
            "add",
        ),
        "combine_derived_max_minus_standing_vertical": (
            "combine_drill_max_vertical_leap",
            "combine_drill_standing_vertical_leap",
            "subtract",
        ),
    }
    for feature, (left, right, operation) in physical_formulas.items():
        if left not in result or right not in result:
            result[feature] = np.nan
        elif operation == "add":
            result[feature] = result[left] + result[right]
        else:
            result[feature] = result[left] - result[right]

    height = result.get(
        "combine_anthro_height_wo_shoes",
        pd.Series(np.nan, index=result.index),
    )
    weight = result.get(
        "combine_anthro_weight", pd.Series(np.nan, index=result.index)
    )
    result["combine_derived_bmi"] = _safe_divide(703.0 * weight, height.pow(2))

    protocol_prefixes = {
        "spot_college": "combine_spot_college_",
        "spot_nba": "combine_spot_nba_",
        "spot_fifteen": "combine_spot_fifteen_",
        "off_dribble_college": "combine_nonstationary_off_drib_college_",
        "off_dribble_fifteen": "combine_nonstationary_off_drib_fifteen_",
        "on_move_college": "combine_nonstationary_on_move_college_",
        "on_move_fifteen": "combine_nonstationary_on_move_fifteen_",
    }
    for protocol, prefix in protocol_prefixes.items():
        made_columns = [
            column
            for column in result.columns
            if column.startswith(prefix) and column.endswith("_made")
        ]
        attempt_columns = [
            column
            for column in result.columns
            if column.startswith(prefix) and column.endswith("_attempt")
        ]
        made_feature = f"combine_shooting_{protocol}_made"
        attempt_feature = f"combine_shooting_{protocol}_attempt"
        if made_columns:
            result[made_feature] = result[made_columns].sum(axis=1, min_count=1)
        else:
            result[made_feature] = np.nan
        if attempt_columns:
            result[attempt_feature] = result[attempt_columns].sum(
                axis=1, min_count=1
            )
        else:
            result[attempt_feature] = np.nan
        result[f"combine_shooting_{protocol}_pct"] = _safe_divide(
            result[made_feature], result[attempt_feature]
        )
    return result


def load_combine_event_names(
    raw_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load combine names by exact NBA player-year key from anthro files."""
    directory = raw_root / "nba" / "combine" / "anthro"
    chunks: list[pd.DataFrame] = []
    for path in sorted(directory.glob("anthro_*.parquet")):
        if "_all" in path.stem:
            continue
        year = parse_year_from_filename(path)
        if not start_year <= year <= end_year:
            continue
        raw = pd.read_parquet(path)
        required = {"PLAYER_ID", "PLAYER_NAME"}
        if not required.issubset(raw.columns):
            raise ValueError(f"{path} is missing {sorted(required - set(raw.columns))}")
        current = pd.DataFrame(
            {
                "nba_player_id": normalize_id(raw["PLAYER_ID"]),
                "draft_year": year,
                "combine_name": raw["PLAYER_NAME"].astype("string"),
            }
        )
        chunks.append(current.dropna(subset=["nba_player_id"]))
    if not chunks:
        raise FileNotFoundError(f"No combine anthro files found in {directory}")

    combined = pd.concat(chunks, ignore_index=True)
    duplicate_mask = combined.duplicated(KEY_COLUMNS, keep=False)
    duplicates = combined.loc[duplicate_mask].sort_values(KEY_COLUMNS)
    name_counts = combined.groupby(KEY_COLUMNS)["combine_name"].nunique(dropna=True)
    conflicts = name_counts[name_counts.gt(1)]
    if not conflicts.empty:
        raise ValueError(
            "Conflicting combine names for exact event keys: "
            f"{conflicts.index.tolist()[:10]}"
        )
    combined = combined.drop_duplicates(KEY_COLUMNS, keep="first")
    parsed = combined["combine_name"].map(parse_combine_name)
    combined[["combine_base_name", "compact_name", "team_qualifier"]] = pd.DataFrame(
        parsed.tolist(), index=combined.index
    )
    return combined.sort_values(KEY_COLUMNS).reset_index(drop=True), duplicates


def load_college_player_team_seasons(
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and collapse derived NCAA rows to player-team-season totals."""
    directory = raw_root / "college" / "player_season_from_box"
    paths = sorted(directory.glob("player_season_from_box_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No college season files found in {directory}")
    chunks = [pd.read_parquet(path) for path in paths]
    raw = pd.concat(chunks, ignore_index=True, sort=False)
    required = {"season", "athlete_id", "athlete_display_name", "team_id"}
    if not required.issubset(raw.columns):
        raise ValueError(
            "College player-season data is missing "
            f"{sorted(required - set(raw.columns))}"
        )

    raw["athlete_id"] = normalize_id(raw["athlete_id"])
    raw["team_id"] = normalize_id(raw["team_id"])
    raw["season"] = pd.to_numeric(raw["season"], errors="coerce").astype("Int64")
    invalid = raw.loc[
        raw[["athlete_id", "season"]].isna().any(axis=1),
        [
            column
            for column in [
                "athlete_id",
                "season",
                "athlete_display_name",
                "team_id",
                "team_display_name",
            ]
            if column in raw.columns
        ],
    ].copy()
    raw = raw.dropna(subset=["athlete_id", "season"]).copy()

    for column in PLAYER_TOTAL_COLUMNS:
        if column not in raw:
            raw[column] = np.nan
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["athlete_position_abbreviation"] = raw.get(
        "athlete_position_abbreviation",
        pd.Series(pd.NA, index=raw.index, dtype="string"),
    ).map(_normalize_college_position)
    raw["team_display_name"] = raw.get(
        "team_display_name",
        pd.Series(pd.NA, index=raw.index, dtype="string"),
    ).astype("string")

    group_columns = ["athlete_id", "season", "team_id"]
    identity = (
        raw.sort_values(group_columns)
        .groupby(group_columns, as_index=False, dropna=False)
        .agg(
            athlete_display_name=("athlete_display_name", "first"),
            team_display_name=("team_display_name", "first"),
            college_position=("athlete_position_abbreviation", "first"),
        )
    )
    totals = (
        raw.groupby(group_columns, as_index=False, dropna=False)[PLAYER_TOTAL_COLUMNS]
        .sum(min_count=1)
    )
    team_rows = identity.merge(totals, on=group_columns, validate="one_to_one")
    team_rows["compact_name"] = team_rows["athlete_display_name"].map(compact_name)
    team_rows["compact_team_name"] = team_rows["team_display_name"].map(compact_name)
    return team_rows, invalid


def build_college_player_seasons(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse transfer rows and retain the highest-minute primary team."""
    group_columns = ["athlete_id", "season"]
    totals = (
        team_rows.groupby(group_columns, as_index=False)[PLAYER_TOTAL_COLUMNS]
        .sum(min_count=1)
    )
    primary = team_rows.assign(
        _minutes_sort=team_rows["minutes"].fillna(-1),
        _games_sort=team_rows["games_played"].fillna(-1),
        _team_sort=team_rows["team_id"].fillna("~"),
    ).sort_values(
        group_columns + ["_minutes_sort", "_games_sort", "_team_sort"],
        ascending=[True, True, False, False, True],
    )
    primary = primary.drop_duplicates(group_columns, keep="first")[
        group_columns
        + [
            "athlete_display_name",
            "compact_name",
            "college_position",
            "team_id",
            "team_display_name",
        ]
    ].rename(
        columns={
            "team_id": "primary_team_id",
            "team_display_name": "primary_team_name",
        }
    )
    seasons = primary.merge(totals, on=group_columns, validate="one_to_one")
    return add_player_rate_features(seasons)


def _candidate_description(candidate_rows: pd.DataFrame) -> str:
    """Return a compact human-auditable description of identity candidates."""
    if candidate_rows.empty:
        return ""
    descriptions: list[str] = []
    for athlete_id, group in candidate_rows.groupby("athlete_id", sort=True):
        names = sorted(group["athlete_display_name"].dropna().astype(str).unique())
        teams = sorted(group["team_display_name"].dropna().astype(str).unique())
        descriptions.append(
            f"{athlete_id}:{' / '.join(names)}:{' / '.join(teams)}"
        )
    return " | ".join(descriptions)


def match_college_identities(
    events: pd.DataFrame,
    college_team_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve exact names without consulting the binary draft outcome."""
    required_event_columns = set(KEY_COLUMNS + ["combine_name"])
    if not required_event_columns.issubset(events.columns):
        raise ValueError(
            f"Identity events are missing {sorted(required_event_columns - set(events))}"
        )
    if "drafted" in events.columns:
        raise ValueError("Identity matching must not receive the drafted target")

    prepared = events[KEY_COLUMNS + ["combine_name"]].copy()
    parsed = prepared["combine_name"].map(parse_combine_name)
    prepared[["combine_base_name", "compact_name", "team_qualifier"]] = pd.DataFrame(
        parsed.tolist(), index=prepared.index
    )

    same_season_groups = {
        key: group
        for key, group in college_team_rows.groupby(
            ["season", "compact_name"], sort=False
        )
    }
    history_groups = {
        name: group
        for name, group in college_team_rows.groupby("compact_name", sort=False)
    }

    match_rows: list[dict[str, object]] = []
    for event in prepared.itertuples(index=False):
        key_values = {
            "nba_player_id": event.nba_player_id,
            "draft_year": int(event.draft_year),
            "combine_name": event.combine_name,
            "combine_base_name": event.combine_base_name,
            "compact_name": event.compact_name,
            "team_qualifier": event.team_qualifier,
        }
        matched_id: object = pd.NA
        method: object = pd.NA
        matched_name: object = pd.NA
        qualifier_alias: object = pd.NA
        status = "unmatched_no_exact_candidate"
        candidates = pd.DataFrame(columns=college_team_rows.columns)

        if event.compact_name:
            candidates = same_season_groups.get(
                (int(event.draft_year), event.compact_name),
                pd.DataFrame(columns=college_team_rows.columns),
            )
            candidate_ids = sorted(
                candidates["athlete_id"].dropna().astype(str).unique()
            )
            if len(candidate_ids) == 1:
                matched_id = candidate_ids[0]
                method = "same_season_unique_compact_name"
                status = "matched"
            elif len(candidate_ids) > 1:
                status = "ambiguous_same_season_compact_name"
                aliases = TEAM_QUALIFIER_ALIASES.get(event.team_qualifier, ())
                if aliases:
                    qualifier_mask = candidates["compact_team_name"].map(
                        lambda team: any(alias in team for alias in aliases)
                    )
                    qualified = candidates.loc[qualifier_mask]
                    qualified_ids = sorted(
                        qualified["athlete_id"].dropna().astype(str).unique()
                    )
                    if len(qualified_ids) == 1:
                        matched_id = qualified_ids[0]
                        method = "same_season_unique_team_qualifier"
                        qualifier_alias = " | ".join(aliases)
                        status = "matched"
                    elif len(qualified_ids) == 0:
                        status = "qualifier_did_not_identify_candidate"
                    else:
                        status = "ambiguous_team_qualifier"
            else:
                history = history_groups.get(
                    event.compact_name,
                    pd.DataFrame(columns=college_team_rows.columns),
                )
                candidates = history.loc[
                    history["season"].between(
                        int(event.draft_year) - 6,
                        int(event.draft_year) - 1,
                    )
                ]
                candidate_ids = sorted(
                    candidates["athlete_id"].dropna().astype(str).unique()
                )
                if len(candidate_ids) == 1:
                    matched_id = candidate_ids[0]
                    method = "prior_six_seasons_unique_compact_name"
                    status = "matched"
                elif len(candidate_ids) > 1:
                    status = "ambiguous_prior_six_seasons_compact_name"

        if pd.notna(matched_id):
            names = candidates.loc[
                candidates["athlete_id"].astype("string").eq(str(matched_id)),
                "athlete_display_name",
            ].dropna()
            matched_name = names.iloc[0] if not names.empty else pd.NA

        match_rows.append(
            {
                **key_values,
                "college_athlete_id": matched_id,
                "college_athlete_name": matched_name,
                "identity_match_method": method,
                "identity_match_status": status,
                "qualifier_alias_used": qualifier_alias,
                "candidate_athlete_count": int(
                    candidates["athlete_id"].nunique(dropna=True)
                ),
                "candidate_description": _candidate_description(candidates),
            }
        )

    result = pd.DataFrame(match_rows)
    result["college_athlete_id"] = result["college_athlete_id"].astype("string")
    if result.duplicated(KEY_COLUMNS).any():
        raise AssertionError("Identity matcher produced duplicate event keys")
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def load_player_core(raw_root: Path) -> pd.DataFrame:
    """Load only pre-draft-safe bio columns from ESPN college core files."""
    directory = raw_root / "college" / "player_core"
    paths = sorted(directory.glob("player_core_*.parquet"))
    if not paths:
        return pd.DataFrame(
            columns=["athlete_id", "season", "date_of_birth", "core_position"]
        )
    chunks: list[pd.DataFrame] = []
    for path in paths:
        raw = pd.read_parquet(path)
        current = pd.DataFrame(
            {
                "athlete_id": normalize_id(raw["athlete_id"]),
                "season": pd.to_numeric(raw["season"], errors="coerce").astype(
                    "Int64"
                ),
                "date_of_birth": raw.get(
                    "date_of_birth", pd.Series(pd.NA, index=raw.index)
                ),
                "core_position": raw.get(
                    "position_abbreviation", pd.Series(pd.NA, index=raw.index)
                ).map(_normalize_college_position),
            }
        )
        chunks.append(current)
    return pd.concat(chunks, ignore_index=True).dropna(
        subset=["athlete_id", "season"]
    )


def build_college_history_features(
    matches: pd.DataFrame,
    player_seasons: pd.DataFrame,
    player_core: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create last-season, career, development, age, and context join keys."""
    matched = matches.dropna(subset=["college_athlete_id"])[
        KEY_COLUMNS + ["college_athlete_id"]
    ]
    history = matched.merge(
        player_seasons,
        left_on="college_athlete_id",
        right_on="athlete_id",
        how="left",
        validate="many_to_many",
    )
    history = history.loc[
        history["season"].notna()
        & history["season"].ge(history["draft_year"] - 6)
        & history["season"].le(history["draft_year"])
    ].copy()

    result = matches[KEY_COLUMNS].copy()
    if history.empty:
        result["has_college_data"] = np.int8(0)
        return result, pd.DataFrame(), pd.DataFrame()

    history = history.sort_values(KEY_COLUMNS + ["season"])
    last = history.groupby(KEY_COLUMNS, as_index=False).tail(1).copy()
    last["college_last_season_gap"] = (
        last["draft_year"].astype(float) - last["season"].astype(float)
    )
    last_features = last[
        KEY_COLUMNS
        + ["college_last_season_gap"]
        + PLAYER_TOTAL_COLUMNS
        + PLAYER_RATE_COLUMNS
    ].rename(
        columns={
            column: f"college_last_{column}"
            for column in PLAYER_TOTAL_COLUMNS + PLAYER_RATE_COLUMNS
        }
    )
    last_features["college_position"] = last["college_position"].map(
        _normalize_college_position
    )

    grouped = history.groupby(KEY_COLUMNS, as_index=False)
    career_totals = grouped[PLAYER_TOTAL_COLUMNS].sum(min_count=1)
    career_meta = grouped.agg(
        college_career_seasons=("season", "nunique"),
        _career_first_season=("season", "min"),
        _career_last_season=("season", "max"),
    )
    career = career_meta.merge(career_totals, on=KEY_COLUMNS, validate="one_to_one")
    career["college_career_span_years"] = (
        career["_career_last_season"] - career["_career_first_season"] + 1
    ).astype(float)
    career = add_player_rate_features(career)
    career = career[
        KEY_COLUMNS
        + ["college_career_seasons", "college_career_span_years"]
        + PLAYER_TOTAL_COLUMNS
        + PLAYER_RATE_COLUMNS
    ].rename(
        columns={
            column: f"college_career_{column}"
            for column in PLAYER_TOTAL_COLUMNS + PLAYER_RATE_COLUMNS
        }
    )

    descending = history.sort_values(
        KEY_COLUMNS + ["season"], ascending=[True, True, False]
    ).copy()
    descending["_history_order"] = descending.groupby(KEY_COLUMNS).cumcount()
    previous = descending.loc[descending["_history_order"].eq(1)].copy()
    development = last[KEY_COLUMNS + ["season"] + DEVELOPMENT_RATE_COLUMNS].merge(
        previous[KEY_COLUMNS + ["season"] + DEVELOPMENT_RATE_COLUMNS],
        on=KEY_COLUMNS,
        how="left",
        suffixes=("_last", "_previous"),
        validate="one_to_one",
    )
    development["college_development_season_gap"] = (
        development["season_last"] - development["season_previous"]
    ).astype(float)
    for column in DEVELOPMENT_RATE_COLUMNS:
        development[f"college_development_{column}_delta"] = (
            development[f"{column}_last"]
            - development[f"{column}_previous"]
        )
    development_columns = [
        "college_development_season_gap",
        *[
            f"college_development_{column}_delta"
            for column in DEVELOPMENT_RATE_COLUMNS
        ],
    ]
    development = development[KEY_COLUMNS + development_columns]

    age_features = matches[KEY_COLUMNS].copy()
    invalid_ages = pd.DataFrame()
    if not player_core.empty:
        core_history = matched.merge(
            player_core,
            left_on="college_athlete_id",
            right_on="athlete_id",
            how="left",
            validate="many_to_many",
        )
        core_history = core_history.loc[
            core_history["season"].notna()
            & core_history["season"].ge(core_history["draft_year"] - 6)
            & core_history["season"].le(core_history["draft_year"])
        ].copy()
        core_history["_has_dob"] = core_history["date_of_birth"].notna()
        bio = (
            core_history.sort_values(
                KEY_COLUMNS + ["_has_dob", "season"],
                ascending=[True, True, False, False],
            )
            .drop_duplicates(KEY_COLUMNS, keep="first")
            .copy()
        )
        bio["date_of_birth"] = pd.to_datetime(
            bio["date_of_birth"], errors="coerce", utc=True
        )
        reference_date = pd.to_datetime(
            bio["draft_year"].astype("Int64").astype("string") + "-07-01",
            errors="coerce",
            utc=True,
        )
        bio["college_age_on_july_1"] = (
            reference_date - bio["date_of_birth"]
        ).dt.days / 365.2425
        invalid_mask = bio["college_age_on_july_1"].notna() & ~bio[
            "college_age_on_july_1"
        ].between(16, 35)
        invalid_ages = bio.loc[
            invalid_mask,
            KEY_COLUMNS
            + [
                "college_athlete_id",
                "season",
                "date_of_birth",
                "college_age_on_july_1",
            ],
        ].copy()
        bio.loc[invalid_mask, "college_age_on_july_1"] = np.nan
        age_features = age_features.merge(
            bio[KEY_COLUMNS + ["college_age_on_july_1"]],
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        core_positions = bio[KEY_COLUMNS + ["core_position"]]
        last_features = last_features.merge(
            core_positions, on=KEY_COLUMNS, how="left", validate="one_to_one"
        )
        last_features["college_position"] = last_features[
            "college_position"
        ].fillna(last_features["core_position"])
        last_features = last_features.drop(columns="core_position")
    else:
        age_features["college_age_on_july_1"] = np.nan

    result = (
        result.merge(last_features, on=KEY_COLUMNS, how="left", validate="one_to_one")
        .merge(career, on=KEY_COLUMNS, how="left", validate="one_to_one")
        .merge(development, on=KEY_COLUMNS, how="left", validate="one_to_one")
        .merge(age_features, on=KEY_COLUMNS, how="left", validate="one_to_one")
    )
    result["has_college_data"] = result["college_last_season_gap"].notna().astype(
        "int8"
    )
    result["college_position"] = result["college_position"].astype("string")

    context_keys = last[
        KEY_COLUMNS
        + [
            "college_athlete_id",
            "season",
            "primary_team_id",
            "primary_team_name",
        ]
    ].rename(columns={"season": "college_last_season"})
    return result, context_keys, invalid_ages


def load_college_team_features(raw_root: Path) -> pd.DataFrame:
    """Pivot stable ESPN team stats and derive efficiency/style context."""
    directory = raw_root / "college" / "team_stats"
    paths = sorted(directory.glob("team_stats_*.parquet"))
    if not paths:
        return pd.DataFrame(columns=["season", "team_id"])
    chunks: list[pd.DataFrame] = []
    for path in paths:
        raw = pd.read_parquet(path, columns=["season", "team_id", "stat_name", "value"])
        chunks.append(raw.loc[raw["stat_name"].isin(TEAM_STAT_COLUMNS)])
    long = pd.concat(chunks, ignore_index=True)
    long["season"] = pd.to_numeric(long["season"], errors="coerce").astype("Int64")
    long["team_id"] = normalize_id(long["team_id"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["season", "team_id"])

    pivot = long.pivot_table(
        index=["season", "team_id"],
        columns="stat_name",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    for source in TEAM_STAT_COLUMNS:
        if source not in pivot:
            pivot[source] = np.nan
    team = pivot[["season", "team_id"] + list(TEAM_STAT_COLUMNS)].rename(
        columns={
            source: f"college_team_{destination}"
            for source, destination in TEAM_STAT_COLUMNS.items()
        }
    )

    def team_column(name: str) -> pd.Series:
        return team[f"college_team_{name}"]

    team["college_team_estimated_possessions"] = (
        team_column("avg_field_goals_attempted")
        - team_column("avg_offensive_rebounds")
        + team_column("avg_turnovers")
        + 0.44 * team_column("avg_free_throws_attempted")
    )
    team["college_team_offensive_rating_est"] = _safe_divide(
        100.0 * team_column("avg_points"),
        team["college_team_estimated_possessions"],
    )
    team["college_team_effective_fg_pct"] = _safe_divide(
        team_column("avg_field_goals_made")
        + 0.5 * team_column("avg_three_point_field_goals_made"),
        team_column("avg_field_goals_attempted"),
    )
    team["college_team_true_shooting_pct"] = _safe_divide(
        team_column("avg_points"),
        2
        * (
            team_column("avg_field_goals_attempted")
            + 0.44 * team_column("avg_free_throws_attempted")
        ),
    )
    team["college_team_three_attempt_rate"] = _safe_divide(
        team_column("avg_three_point_field_goals_attempted"),
        team_column("avg_field_goals_attempted"),
    )
    team["college_team_free_throw_rate"] = _safe_divide(
        team_column("avg_free_throws_attempted"),
        team_column("avg_field_goals_attempted"),
    )
    team["college_team_turnover_rate"] = _safe_divide(
        team_column("avg_turnovers"),
        team["college_team_estimated_possessions"],
    )
    team["college_team_assist_per_field_goal"] = _safe_divide(
        team_column("avg_assists"), team_column("avg_field_goals_made")
    )

    for feature in TEAM_PERCENTILE_FEATURES:
        source = f"college_team_{feature}"
        team[f"{source}_percentile"] = team.groupby("season")[source].rank(
            method="average", pct=True
        )
    return team.sort_values(["season", "team_id"]).reset_index(drop=True)


def join_team_context(
    history_features: pd.DataFrame,
    context_keys: pd.DataFrame,
    team_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join latest primary-team context and return an identity-bearing audit."""
    if context_keys.empty or team_features.empty:
        return history_features, context_keys.copy()
    context = context_keys.merge(
        team_features,
        left_on=["college_last_season", "primary_team_id"],
        right_on=["season", "team_id"],
        how="left",
        validate="many_to_one",
    )
    team_columns = [
        column for column in team_features if column.startswith("college_team_")
    ]
    model_context = context[KEY_COLUMNS + team_columns]
    enriched = history_features.merge(
        model_context, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    context["team_context_available"] = context[team_columns].notna().any(axis=1)
    audit_columns = (
        KEY_COLUMNS
        + [
            "college_athlete_id",
            "college_last_season",
            "primary_team_id",
            "primary_team_name",
            "team_context_available",
        ]
    )
    return enriched, context[audit_columns]


def build_feature_missingness_audit(
    dataset: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, group in dataset.groupby("draft_year", sort=True):
        for feature in feature_columns:
            missing = int(group[feature].isna().sum())
            rows.append(
                {
                    "draft_year": int(year),
                    "feature": feature,
                    "feature_type": (
                        "categorical"
                        if not is_numeric_dtype(dataset[feature])
                        else "numeric"
                    ),
                    "rows": len(group),
                    "missing_values": missing,
                    "missing_rate": missing / max(len(group), 1),
                }
            )
    return pd.DataFrame(rows)


def build_feature_manifest(dataset: pd.DataFrame) -> pd.DataFrame:
    """Describe target-safe model columns by ablation-friendly group."""
    rows: list[dict[str, object]] = []
    for column in dataset.columns:
        if column in KEY_COLUMNS:
            group, role = "event_key", "identifier"
        elif column == "drafted":
            group, role = "target", "target"
        elif column.startswith("combine_derived_"):
            group, role = "combine_derived", "feature"
        elif column.startswith("combine_shooting_"):
            group, role = "combine_shooting", "feature"
        elif column.startswith("combine_spot_") or column.startswith(
            "combine_nonstationary_"
        ):
            group, role = "combine_shooting_raw", "feature"
        elif column == "position" or column.startswith("combine_"):
            group, role = "combine", "feature"
        elif column == "has_college_data":
            group, role = "college_availability", "feature"
        elif column.startswith("college_last_"):
            group, role = "college_last_season", "feature"
        elif column.startswith("college_career_"):
            group, role = "college_career", "feature"
        elif column.startswith("college_development_"):
            group, role = "college_development", "feature"
        elif column.startswith("college_team_"):
            group, role = "college_team_context", "feature"
        elif column in {"college_position", "college_age_on_july_1"}:
            group, role = "college_bio", "feature"
        else:
            group, role = "other", "feature"
        rows.append(
            {
                "column": column,
                "role": role,
                "feature_group": group,
                "dtype": str(dataset[column].dtype),
            }
        )
    return pd.DataFrame(rows)


def _validate_final_dataset(dataset: pd.DataFrame) -> None:
    if dataset.duplicated(KEY_COLUMNS).any():
        raise AssertionError("Enriched dataset event key is not unique")
    if dataset[KEY_COLUMNS].isna().any(axis=None):
        raise AssertionError("Enriched dataset contains a null event key")
    if not set(dataset["drafted"].dropna().unique()).issubset({0, 1}):
        raise AssertionError("drafted must remain binary")

    forbidden_exact = {
        "college_athlete_id",
        "athlete_id",
        "team_id",
        "primary_team_id",
        "identity_match_method",
        "identity_match_status",
        "combine_name",
        "combine_base_name",
        "college_athlete_name",
        "team_display_name",
        "primary_team_name",
        "date_of_birth",
    }
    leaked = forbidden_exact.intersection(dataset.columns)
    unexpected_ids = {
        column
        for column in dataset.columns
        if column.endswith("_id") and column != "nba_player_id"
    }
    if leaked or unexpected_ids:
        raise AssertionError(
            "Identity fields reached the model table: "
            f"{sorted(leaked | unexpected_ids)}"
        )
    non_numeric = [
        column
        for column in dataset.columns
        if column not in KEY_COLUMNS + ["position", "college_position"]
        and not is_numeric_dtype(dataset[column])
    ]
    if non_numeric:
        raise AssertionError(f"Unexpected non-numeric model columns: {non_numeric}")


def build_enriched_draft_classification_dataset(
    raw_root: Path,
    processed_root: Path,
    start_year: int = 2009,
    end_year: int = 2022,
    base_dataset_path: Path | None = None,
    output_path: Path | None = None,
    audit_dir: Path | None = None,
) -> pd.DataFrame:
    """Build, validate, audit, save, and return the enriched model table."""
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    base_dataset_path = (
        Path(base_dataset_path)
        if base_dataset_path is not None
        else processed_root / "draft_classification_dataset.parquet"
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else processed_root / "draft_classification_enriched_dataset.parquet"
    )
    audit_dir = (
        Path(audit_dir)
        if audit_dir is not None
        else processed_root / "audit" / "draft_classification_enriched"
    )
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not base_dataset_path.exists():
        build_draft_classification_dataset(
            raw_root=raw_root,
            processed_root=processed_root,
            start_year=start_year,
            end_year=end_year,
            output_path=base_dataset_path,
        )
    base = pd.read_parquet(base_dataset_path)
    base = base.loc[base["draft_year"].between(start_year, end_year)].copy()
    required = set(KEY_COLUMNS + ["position", "drafted"])
    if not required.issubset(base.columns):
        raise ValueError(
            f"Base classifier dataset is missing {sorted(required - set(base))}"
        )
    _validate_final_dataset(base)

    # Verify that enriching did not inherit a stale or modified target.
    official_draft_keys, _, _ = load_draft_keys(raw_root, start_year, end_year)
    official_index = pd.MultiIndex.from_frame(official_draft_keys[KEY_COLUMNS])
    base_index = pd.MultiIndex.from_frame(base[KEY_COLUMNS])
    expected_target = base_index.isin(official_index).astype("int8")
    if not np.array_equal(base["drafted"].to_numpy(), expected_target):
        raise AssertionError("Base drafted target does not match official exact keys")

    print("Loading combine event names and NCAA player seasons...")
    combine_names, combine_name_duplicates = load_combine_event_names(
        raw_root, start_year, end_year
    )
    _write_csv(
        combine_name_duplicates,
        audit_dir / "combine_name_duplicate_key_rows.csv",
        KEY_COLUMNS + ["combine_name"],
    )
    events = base[KEY_COLUMNS].merge(
        combine_names[KEY_COLUMNS + ["combine_name"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    missing_combine_names = events.loc[events["combine_name"].isna()].copy()
    _write_csv(
        missing_combine_names,
        audit_dir / "events_missing_combine_name.csv",
        KEY_COLUMNS + ["combine_name"],
    )

    college_team_rows, invalid_college_rows = load_college_player_team_seasons(
        raw_root
    )
    _write_csv(
        invalid_college_rows,
        audit_dir / "college_rows_missing_identity_key.csv",
    )
    player_seasons = build_college_player_seasons(college_team_rows)

    print("Matching exact, target-independent NCAA identities...")
    matches = match_college_identities(events, college_team_rows)
    match_audit = matches.merge(
        base[KEY_COLUMNS + ["drafted"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    _write_csv(match_audit, audit_dir / "college_identity_matches_all.csv")
    _write_csv(
        match_audit.loc[match_audit["college_athlete_id"].isna()],
        audit_dir / "college_identity_unmatched_or_ambiguous.csv",
    )
    _write_csv(
        match_audit.loc[
            match_audit["identity_match_status"].astype("string").str.startswith(
                "ambiguous", na=False
            )
            | match_audit["identity_match_status"].astype("string").str.startswith(
                "qualifier", na=False
            )
        ],
        audit_dir / "college_identity_ambiguous.csv",
    )

    print("Building player history, development, age, and team context...")
    player_core = load_player_core(raw_root)
    history_features, context_keys, invalid_ages = build_college_history_features(
        matches, player_seasons, player_core
    )
    _write_csv(
        invalid_ages,
        audit_dir / "implausible_college_ages_excluded.csv",
        KEY_COLUMNS
        + [
            "college_athlete_id",
            "season",
            "date_of_birth",
            "college_age_on_july_1",
        ],
    )
    team_features = load_college_team_features(raw_root)
    history_features, team_context_audit = join_team_context(
        history_features, context_keys, team_features
    )
    _write_csv(
        team_context_audit,
        audit_dir / "college_primary_team_context.csv",
        KEY_COLUMNS
        + [
            "college_athlete_id",
            "college_last_season",
            "primary_team_id",
            "primary_team_name",
            "team_context_available",
        ],
    )

    target = base[KEY_COLUMNS + ["drafted"]]
    predictors = add_combine_derived_features(base.drop(columns="drafted"))
    enriched = (
        predictors.merge(
            history_features, on=KEY_COLUMNS, how="left", validate="one_to_one"
        )
        .merge(target, on=KEY_COLUMNS, how="left", validate="one_to_one")
    )
    enriched["has_college_data"] = enriched["has_college_data"].fillna(0).astype(
        "int8"
    )
    enriched["college_position"] = enriched["college_position"].astype("string")
    enriched = enriched.sort_values(KEY_COLUMNS).reset_index(drop=True)
    _validate_final_dataset(enriched)
    enriched.to_parquet(output_path, index=False)

    match_coverage = match_audit.assign(
        college_identity_matched=match_audit["college_athlete_id"].notna()
    ).merge(
        enriched[
            KEY_COLUMNS
            + ["has_college_data", "college_age_on_july_1"]
        ],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    team_available = (
        team_context_audit[KEY_COLUMNS + ["team_context_available"]]
        if not team_context_audit.empty
        else pd.DataFrame(columns=KEY_COLUMNS + ["team_context_available"])
    )
    match_coverage = match_coverage.merge(
        team_available, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    coverage_by_year_target = (
        match_coverage.groupby(["draft_year", "drafted"], as_index=False)
        .agg(
            rows=("nba_player_id", "size"),
            identity_matches=("college_identity_matched", "sum"),
            rows_with_college_data=("has_college_data", "sum"),
            rows_with_age=("college_age_on_july_1", "count"),
            rows_with_team_context=("team_context_available", "sum"),
        )
    )
    for count_column in [
        "identity_matches",
        "rows_with_college_data",
        "rows_with_age",
        "rows_with_team_context",
    ]:
        coverage_by_year_target[f"{count_column}_rate"] = (
            coverage_by_year_target[count_column] / coverage_by_year_target["rows"]
        )
    _write_csv(
        coverage_by_year_target,
        audit_dir / "college_coverage_by_year_and_target.csv",
    )

    method_summary = (
        match_audit.assign(
            identity_match_method=match_audit["identity_match_method"].fillna(
                "unmatched"
            )
        )
        .groupby("identity_match_method", as_index=False)
        .agg(rows=("nba_player_id", "size"), drafted=("drafted", "sum"))
    )
    method_summary["drafted_rate"] = method_summary["drafted"] / method_summary[
        "rows"
    ]
    _write_csv(method_summary, audit_dir / "college_match_method_summary.csv")

    feature_columns = [
        column
        for column in enriched.columns
        if column not in KEY_COLUMNS + ["drafted"]
    ]
    missingness = build_feature_missingness_audit(enriched, feature_columns)
    _write_csv(
        missingness,
        audit_dir / "enriched_feature_missingness_by_year.csv",
    )
    manifest = build_feature_manifest(enriched)
    _write_csv(manifest, audit_dir / "feature_manifest.csv")

    dataset_summary = pd.DataFrame(
        [
            {
                "start_year": start_year,
                "end_year": end_year,
                "player_year_rows": len(enriched),
                "unique_player_ids": enriched["nba_player_id"].nunique(),
                "drafted": int(enriched["drafted"].sum()),
                "undrafted": int(enriched["drafted"].eq(0).sum()),
                "college_identity_matches": int(
                    match_audit["college_athlete_id"].notna().sum()
                ),
                "rows_with_college_data": int(enriched["has_college_data"].sum()),
                "rows_with_age": int(enriched["college_age_on_july_1"].notna().sum()),
                "rows_with_team_context": int(
                    match_coverage["team_context_available"].fillna(False).sum()
                ),
                "combine_features": int(
                    sum(
                        column.startswith("combine_")
                        and not column.startswith("combine_derived_")
                        and not column.startswith("combine_shooting_")
                        for column in enriched
                    )
                ),
                "combine_derived_features": int(
                    sum(column.startswith("combine_derived_") for column in enriched)
                ),
                "combine_shooting_features": int(
                    sum(column.startswith("combine_shooting_") for column in enriched)
                ),
                "college_last_features": int(
                    sum(column.startswith("college_last_") for column in enriched)
                ),
                "college_career_features": int(
                    sum(column.startswith("college_career_") for column in enriched)
                ),
                "college_development_features": int(
                    sum(
                        column.startswith("college_development_")
                        for column in enriched
                    )
                ),
                "college_team_features": int(
                    sum(column.startswith("college_team_") for column in enriched)
                ),
                "total_model_features": len(feature_columns),
            }
        ]
    )
    _write_csv(dataset_summary, audit_dir / "dataset_summary.csv")

    print("\nCollege identity match methods")
    print(method_summary.to_string(index=False))
    print("\nEnriched dataset summary")
    print(dataset_summary.to_string(index=False))
    print(f"Saved: {output_path}")
    print(f"Audit CSVs: {audit_dir}")
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a target-independent combine + NCAA feature table for the "
            "drafted/not-drafted XGBoost classifier."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--draft-start", type=int, default=2009)
    parser.add_argument("--draft-end", type=int, default=2022)
    parser.add_argument("--base-dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_enriched_draft_classification_dataset(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        start_year=args.draft_start,
        end_year=args.draft_end,
        base_dataset_path=args.base_dataset,
        output_path=args.output,
        audit_dir=args.audit_dir,
    )


if __name__ == "__main__":
    main()
