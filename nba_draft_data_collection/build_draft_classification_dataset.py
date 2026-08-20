"""Build a post-combine dataset for classifying whether a prospect is drafted.

The modeling population is conditional on NBA Draft Combine participation.  A
row represents one normalized NBA ``PLAYER_ID`` in one combine/draft year and
is retained only when that key appears in all four collected combine sources:
anthropometrics, strength/agility drills, spot shooting, and non-stationary
shooting.  The binary ``drafted`` target is a direct exact-key comparison with
official NBA Draft History ``PERSON_ID`` in the same year; names and fuzzy
identity matching are never used.

This is a *post-combine* forecast, not a model for every NCAA player and not a
pre-invitation scouting model.  Combine invitation/participation already
selects a strong prospect pool.  In addition, the shooting endpoints can
contain roster rows whose measurements are partly or entirely missing,
especially in some seasons.  Those values remain null for downstream
fold-specific imputation.  This builder intentionally does not create
missingness/availability flags, because they can encode data-collection era
rather than player ability.

The saved table contains combine measurements, the categorical ``position``,
the key columns, and ``drafted`` only.  Player names, temporary IDs, draft pick
or round, team/organization, and all post-draft NBA outcomes are excluded.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas.api.types import is_numeric_dtype

try:  # Support both ``python -m ...`` and direct script execution.
    from .build_model_dataset_v3 import normalize_id, parse_year_from_filename
except ImportError:  # pragma: no cover - used by direct script execution.
    from build_model_dataset_v3 import normalize_id, parse_year_from_filename


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = SCRIPT_DIR / "data" / "raw"
DEFAULT_PROCESSED_ROOT = SCRIPT_DIR / "data" / "processed"

# Folder, audit/source label, and output feature prefix.
COMBINE_SOURCES = (
    ("anthro", "anthro", "combine_anthro_"),
    ("drills", "drills", "combine_drill_"),
    ("spot_shooting", "spot_shooting", "combine_spot_"),
    (
        "nonstationary_shooting",
        "nonstationary_shooting",
        "combine_nonstationary_",
    ),
)

IDENTITY_OR_TEXT_COLUMNS = {
    "first_name",
    "last_name",
    "player_name",
    "player",
    "name",
    "temp_player_id",
    "temporary_player_id",
    "player_profile_flag",
}

DRAFT_OUTCOME_COLUMNS = {
    "drafted",
    "draft_status",
    "draft_round",
    "draft_round_number",
    "draft_round_pick",
    "draft_pick",
    "overall_pick",
    "round_number",
    "round_pick",
    "team_abbreviation",
    "organization",
    "organization_type",
}

KEY_COLUMNS = ["nba_player_id", "draft_year"]


@dataclass
class CombineSourceResult:
    """A cleaned combine source plus the audit tables created while loading."""

    data: pd.DataFrame
    duplicate_rows: pd.DataFrame
    missing_id_rows: pd.DataFrame
    dropped_columns: pd.DataFrame
    normalized_column_collisions: pd.DataFrame
    file_coverage: pd.DataFrame


def normalize_column_name(value: object) -> str:
    """Convert a source column label to deterministic lower snake case."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_position(value: object) -> object:
    """Normalize combine position labels while preserving them as categories."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper().replace("/", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text if text else pd.NA


def _mode_or_first(series: pd.Series) -> object:
    """Choose a deterministic non-null categorical value for duplicate rows."""
    values = series.dropna().astype("string")
    if values.empty:
        return pd.NA
    modes = values.mode(dropna=True)
    if not modes.empty:
        return sorted(modes.astype(str).tolist())[0]
    return str(values.iloc[0])


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    columns_if_empty: list[str] | None = None,
) -> None:
    """Write an audit CSV and preserve informative headers when it is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and len(frame.columns) == 0 and columns_if_empty is not None:
        frame = pd.DataFrame(columns=columns_if_empty)
    frame.to_csv(path, index=False)


def _feature_exclusion_reason(normalized_column: str) -> str | None:
    """Explain why a non-key combine source column cannot be a predictor."""
    if normalized_column == "position":
        return "categorical_position_handled_separately"
    if normalized_column in IDENTITY_OR_TEXT_COLUMNS:
        return "name_or_temporary_identifier"
    if normalized_column.endswith("_id") or normalized_column == "id":
        return "non_key_identifier"
    if normalized_column in {"season", "year", "draft_year"}:
        return "year_parsed_from_filename"
    if normalized_column in DRAFT_OUTCOME_COLUMNS:
        return "draft_outcome_or_post_selection_field"
    if normalized_column.startswith("draft_"):
        return "draft_outcome_or_post_selection_field"
    return None


def _find_raw_column(
    columns: pd.Index,
    normalized_name: str,
) -> object | None:
    """Find the first raw column whose normalized label equals a target name."""
    for column in columns:
        if normalize_column_name(column) == normalized_name:
            return column
    return None


def load_combine_source(
    directory: Path,
    source_name: str,
    feature_prefix: str,
    start_year: int,
    end_year: int,
) -> CombineSourceResult:
    """Load, normalize, type-coerce, audit, and key-collapse one source.

    Years are derived from filenames, never from mutable source columns.
    Duplicate ``PLAYER_ID``/year rows are retained in the audit and collapsed
    with numeric means solely to maintain one modeling row per exact key.
    """
    paths = sorted(
        path
        for path in directory.glob("*.parquet")
        if "_all" not in path.stem
        and start_year <= parse_year_from_filename(path) <= end_year
    )
    if not paths:
        raise FileNotFoundError(
            f"No {source_name} combine files for {start_year}-{end_year} "
            f"were found in {directory}"
        )

    chunks: list[pd.DataFrame] = []
    missing_id_chunks: list[pd.DataFrame] = []
    dropped_column_rows: list[dict[str, object]] = []
    collision_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for path in paths:
        year = parse_year_from_filename(path)
        raw = pd.read_parquet(path).copy()
        player_id_column = _find_raw_column(raw.columns, "player_id")
        if player_id_column is None:
            raise ValueError(
                f"{path} does not contain the required PLAYER_ID column"
            )

        position_column = _find_raw_column(raw.columns, "position")
        cleaned = pd.DataFrame(index=raw.index)
        cleaned["nba_player_id"] = normalize_id(raw[player_id_column])
        cleaned["draft_year"] = year
        cleaned["position"] = (
            raw[position_column].map(normalize_position)
            if position_column is not None
            else pd.Series(pd.NA, index=raw.index, dtype="string")
        )
        cleaned["_source_file"] = path.name

        used_feature_names: dict[str, str] = {}
        numeric_features_in_file: list[str] = []
        for raw_column in raw.columns:
            normalized = normalize_column_name(raw_column)
            if raw_column == player_id_column:
                dropped_column_rows.append(
                    {
                        "source": source_name,
                        "source_file": path.name,
                        "raw_column": str(raw_column),
                        "normalized_column": normalized,
                        "reason": "primary_key_normalized_separately",
                    }
                )
                continue

            exclusion_reason = _feature_exclusion_reason(normalized)
            if exclusion_reason is not None:
                dropped_column_rows.append(
                    {
                        "source": source_name,
                        "source_file": path.name,
                        "raw_column": str(raw_column),
                        "normalized_column": normalized,
                        "reason": exclusion_reason,
                    }
                )
                continue

            feature_name = feature_prefix + normalized
            if feature_name in used_feature_names:
                collision_rows.append(
                    {
                        "source": source_name,
                        "source_file": path.name,
                        "feature_name": feature_name,
                        "kept_raw_column": used_feature_names[feature_name],
                        "dropped_raw_column": str(raw_column),
                    }
                )
                continue

            converted = pd.to_numeric(raw[raw_column], errors="coerce")
            if not converted.notna().any() and not is_numeric_dtype(
                raw[raw_column]
            ):
                dropped_column_rows.append(
                    {
                        "source": source_name,
                        "source_file": path.name,
                        "raw_column": str(raw_column),
                        "normalized_column": normalized,
                        "reason": "non_numeric_measurement",
                    }
                )
                continue

            cleaned[feature_name] = converted.astype(float)
            used_feature_names[feature_name] = str(raw_column)
            numeric_features_in_file.append(feature_name)

        missing_id_mask = cleaned["nba_player_id"].isna()
        if missing_id_mask.any():
            missing_id_chunks.append(
                cleaned.loc[
                    missing_id_mask,
                    ["draft_year", "_source_file"],
                ].assign(source=source_name, reason="missing_player_id")
            )

        valid = cleaned.loc[~missing_id_mask].copy()
        duplicate_count = int(
            valid.duplicated(KEY_COLUMNS, keep=False).sum()
        )
        rows_with_measurement = (
            int(valid[numeric_features_in_file].notna().any(axis=1).sum())
            if numeric_features_in_file
            else 0
        )
        coverage_rows.append(
            {
                "source": source_name,
                "source_file": path.name,
                "draft_year": year,
                "raw_rows": len(raw),
                "rows_with_player_id": len(valid),
                "unique_player_year_keys": valid[KEY_COLUMNS]
                .drop_duplicates()
                .shape[0],
                "duplicate_key_rows": duplicate_count,
                "numeric_measurement_columns": len(
                    numeric_features_in_file
                ),
                "rows_with_any_numeric_measurement": rows_with_measurement,
            }
        )
        chunks.append(valid)

    combined = pd.concat(chunks, ignore_index=True, sort=False)
    duplicate_mask = combined.duplicated(KEY_COLUMNS, keep=False)
    duplicate_columns = [
        "nba_player_id",
        "draft_year",
        "position",
        "_source_file",
    ]
    duplicate_rows = combined.loc[
        duplicate_mask, duplicate_columns
    ].copy()
    if not duplicate_rows.empty:
        duplicate_rows.insert(0, "source", source_name)

    feature_columns = sorted(
        column
        for column in combined.columns
        if column.startswith(feature_prefix)
    )
    globally_empty = [
        column for column in feature_columns if not combined[column].notna().any()
    ]
    for column in globally_empty:
        dropped_column_rows.append(
            {
                "source": source_name,
                "source_file": "<all requested years>",
                "raw_column": pd.NA,
                "normalized_column": column,
                "reason": "all_values_missing_across_requested_years",
            }
        )
    combined = combined.drop(columns=globally_empty)
    feature_columns = [
        column for column in feature_columns if column not in globally_empty
    ]

    aggregations: dict[str, str | Callable[[pd.Series], object]] = {
        column: "mean" for column in feature_columns
    }
    aggregations["position"] = _mode_or_first
    data = (
        combined.groupby(KEY_COLUMNS, as_index=False, dropna=False)
        .agg(aggregations)
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    missing_ids = (
        pd.concat(missing_id_chunks, ignore_index=True, sort=False)
        if missing_id_chunks
        else pd.DataFrame(
            columns=["source", "draft_year", "_source_file", "reason"]
        )
    )
    dropped_columns = pd.DataFrame(
        dropped_column_rows,
        columns=[
            "source",
            "source_file",
            "raw_column",
            "normalized_column",
            "reason",
        ],
    )
    collisions = pd.DataFrame(
        collision_rows,
        columns=[
            "source",
            "source_file",
            "feature_name",
            "kept_raw_column",
            "dropped_raw_column",
        ],
    )
    coverage = pd.DataFrame(coverage_rows).sort_values("draft_year")
    return CombineSourceResult(
        data=data,
        duplicate_rows=duplicate_rows,
        missing_id_rows=missing_ids,
        dropped_columns=dropped_columns,
        normalized_column_collisions=collisions,
        file_coverage=coverage,
    )


def load_draft_keys(
    raw_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load exact official draft keys without importing outcome metadata."""
    draft_dir = raw_root / "nba" / "draft"
    rollup = draft_dir / "draft_all.parquet"
    required_columns = ["PERSON_ID", "SEASON"]

    if rollup.exists():
        draft = pd.read_parquet(rollup, columns=required_columns)
    else:
        paths = sorted(
            path
            for path in draft_dir.glob("draft_*.parquet")
            if path.stem != "draft_all"
            and start_year <= parse_year_from_filename(path) <= end_year
        )
        if not paths:
            raise FileNotFoundError(
                f"No NBA Draft History Parquet files found in {draft_dir}"
            )
        draft = pd.concat(
            [
                pd.read_parquet(path, columns=required_columns)
                for path in paths
            ],
            ignore_index=True,
            sort=False,
        )

    keys = pd.DataFrame(
        {
            "nba_player_id": normalize_id(draft["PERSON_ID"]),
            "draft_year": pd.to_numeric(
                draft["SEASON"], errors="coerce"
            ).astype("Int64"),
        }
    )
    keys = keys[keys["draft_year"].between(start_year, end_year)].copy()

    missing_key_rows = keys[keys[KEY_COLUMNS].isna().any(axis=1)].copy()
    valid = keys.dropna(subset=KEY_COLUMNS).copy()
    duplicate_rows = valid[valid.duplicated(KEY_COLUMNS, keep=False)].copy()
    valid = valid.drop_duplicates(KEY_COLUMNS).sort_values(KEY_COLUMNS)
    return valid.reset_index(drop=True), duplicate_rows, missing_key_rows


def build_key_membership_audit(
    source_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return combine keys that are absent from one or more required sources."""
    pieces = []
    for source_name, frame in source_frames.items():
        piece = frame[KEY_COLUMNS].drop_duplicates().copy()
        piece["source"] = source_name
        piece["present"] = 1
        pieces.append(piece)

    long = pd.concat(pieces, ignore_index=True)
    membership = (
        long.pivot_table(
            index=KEY_COLUMNS,
            columns="source",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for source_name in source_frames:
        if source_name not in membership:
            membership[source_name] = 0
        membership[source_name] = membership[source_name].astype("int8")

    source_columns = list(source_frames)
    membership["sources_present"] = membership[source_columns].sum(axis=1)
    return membership[
        membership["sources_present"].lt(len(source_columns))
    ].sort_values(KEY_COLUMNS)


def merge_combine_sources(
    source_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-merge the four clean source rosters and reconcile position."""
    merged: pd.DataFrame | None = None
    position_columns: list[str] = []
    feature_order: list[str] = []

    for source_name, frame in source_frames.items():
        position_column = f"_position_{source_name}"
        current = frame.rename(columns={"position": position_column}).copy()
        position_columns.append(position_column)
        feature_order.extend(
            column
            for column in current.columns
            if column not in KEY_COLUMNS + [position_column]
        )
        if merged is None:
            merged = current
        else:
            merged = merged.merge(
                current,
                on=KEY_COLUMNS,
                how="inner",
                validate="one_to_one",
            )

    if merged is None:
        raise ValueError("No combine sources were provided")

    normalized_positions = merged[position_columns].apply(
        lambda column: column.map(normalize_position)
    )
    conflict_mask = normalized_positions.nunique(axis=1, dropna=True).gt(1)
    position_conflicts = pd.concat(
        [merged.loc[conflict_mask, KEY_COLUMNS], normalized_positions[conflict_mask]],
        axis=1,
    )
    merged["position"] = normalized_positions.apply(
        lambda row: next(
            (value for value in row if pd.notna(value)), pd.NA
        ),
        axis=1,
    ).astype("string")
    merged = merged.drop(columns=position_columns)

    ordered_features = list(dict.fromkeys(feature_order))
    merged = merged[KEY_COLUMNS + ["position"] + ordered_features]
    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True), position_conflicts


def build_missingness_audit(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Summarize null measurements without adding flags to the model table."""
    rows: list[dict[str, object]] = []
    audited_features = ["position"] + feature_columns
    for year, group in dataset.groupby("draft_year", sort=True):
        for feature in audited_features:
            missing = int(group[feature].isna().sum())
            rows.append(
                {
                    "draft_year": int(year),
                    "feature": feature,
                    "feature_type": (
                        "categorical" if feature == "position" else "numeric"
                    ),
                    "rows": len(group),
                    "missing_values": missing,
                    "missing_rate": missing / max(len(group), 1),
                }
            )
    return pd.DataFrame(rows)


def build_draft_classification_dataset(
    raw_root: Path,
    processed_root: Path,
    start_year: int = 2009,
    end_year: int = 2022,
    output_path: Path | None = None,
    audit_dir: Path | None = None,
) -> pd.DataFrame:
    """Build, audit, save, and return the combine-participant dataset.

    The cohort is the inner intersection of all four combine source key sets.
    ``drafted`` is one exactly when ``(draft_year, nba_player_id)`` occurs in
    official NBA Draft History.  No draft metadata is merged into predictors.
    """
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    output_path = (
        Path(output_path)
        if output_path is not None
        else processed_root / "draft_classification_dataset.parquet"
    )
    audit_dir = (
        Path(audit_dir)
        if audit_dir is not None
        else processed_root / "audit" / "draft_classification"
    )

    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    combine_root = raw_root / "nba" / "combine"
    results: dict[str, CombineSourceResult] = {}
    print("Loading and auditing the four NBA Draft Combine sources...")
    for folder, source_name, feature_prefix in COMBINE_SOURCES:
        results[source_name] = load_combine_source(
            combine_root / folder,
            source_name,
            feature_prefix,
            start_year,
            end_year,
        )

    source_frames = {
        source_name: result.data for source_name, result in results.items()
    }
    duplicate_rows = pd.concat(
        [result.duplicate_rows for result in results.values()],
        ignore_index=True,
        sort=False,
    )
    missing_id_rows = pd.concat(
        [result.missing_id_rows for result in results.values()],
        ignore_index=True,
        sort=False,
    )
    dropped_columns = pd.concat(
        [result.dropped_columns for result in results.values()],
        ignore_index=True,
        sort=False,
    )
    column_collisions = pd.concat(
        [result.normalized_column_collisions for result in results.values()],
        ignore_index=True,
        sort=False,
    )
    file_coverage = pd.concat(
        [result.file_coverage for result in results.values()],
        ignore_index=True,
        sort=False,
    ).sort_values(["draft_year", "source"])

    _write_csv(
        duplicate_rows,
        audit_dir / "combine_duplicate_key_rows.csv",
        ["source", "nba_player_id", "draft_year", "position", "_source_file"],
    )
    _write_csv(
        missing_id_rows,
        audit_dir / "combine_rows_missing_player_id.csv",
        ["source", "draft_year", "_source_file", "reason"],
    )
    _write_csv(
        dropped_columns,
        audit_dir / "dropped_combine_columns.csv",
    )
    _write_csv(
        column_collisions,
        audit_dir / "normalized_column_collisions.csv",
        [
            "source",
            "source_file",
            "feature_name",
            "kept_raw_column",
            "dropped_raw_column",
        ],
    )
    _write_csv(file_coverage, audit_dir / "combine_file_coverage.csv")

    source_key_mismatches = build_key_membership_audit(source_frames)
    _write_csv(
        source_key_mismatches,
        audit_dir / "combine_source_key_mismatches.csv",
        KEY_COLUMNS
        + [source_name for _, source_name, _ in COMBINE_SOURCES]
        + ["sources_present"],
    )

    cohort, position_conflicts = merge_combine_sources(source_frames)
    _write_csv(
        position_conflicts,
        audit_dir / "combine_position_conflicts.csv",
        KEY_COLUMNS
        + [f"_position_{source}" for source in source_frames],
    )

    repeated_ids = (
        cohort.groupby("nba_player_id", as_index=False)
        .agg(
            combine_year_count=("draft_year", "nunique"),
            first_combine_year=("draft_year", "min"),
            last_combine_year=("draft_year", "max"),
            combine_years=(
                "draft_year",
                lambda years: " | ".join(
                    str(int(year)) for year in sorted(years.unique())
                ),
            ),
        )
    )
    repeated_ids = repeated_ids[repeated_ids["combine_year_count"].gt(1)]
    _write_csv(
        repeated_ids,
        audit_dir / "repeated_player_ids_across_combine_years.csv",
        [
            "nba_player_id",
            "combine_year_count",
            "first_combine_year",
            "last_combine_year",
            "combine_years",
        ],
    )

    print("Joining the direct official Draft History target...")
    draft_keys, draft_duplicates, draft_missing_keys = load_draft_keys(
        raw_root, start_year, end_year
    )
    _write_csv(
        draft_duplicates,
        audit_dir / "draft_duplicate_key_rows.csv",
        KEY_COLUMNS,
    )
    _write_csv(
        draft_missing_keys,
        audit_dir / "draft_rows_missing_key.csv",
        KEY_COLUMNS,
    )

    official_draft_index = pd.MultiIndex.from_frame(draft_keys[KEY_COLUMNS])
    cohort_index = pd.MultiIndex.from_frame(cohort[KEY_COLUMNS])
    cohort["drafted"] = cohort_index.isin(official_draft_index).astype("int8")

    if cohort.duplicated(KEY_COLUMNS).any():
        raise AssertionError("Final nba_player_id + draft_year key is not unique")

    feature_columns = [
        column for column in cohort.columns if column.startswith("combine_")
    ]
    non_numeric_features = [
        column
        for column in feature_columns
        if not is_numeric_dtype(cohort[column])
    ]
    if non_numeric_features:
        raise AssertionError(
            "Combine measurements were not numeric-coerced: "
            f"{non_numeric_features}"
        )

    forbidden_columns = (
        IDENTITY_OR_TEXT_COLUMNS
        | DRAFT_OUTCOME_COLUMNS.difference({"drafted"})
        | {"temp_player_id", "person_id"}
    )
    leaked = sorted(forbidden_columns.intersection(cohort.columns))
    unexpected_ids = sorted(
        column
        for column in cohort.columns
        if column.endswith("_id") and column != "nba_player_id"
    )
    if leaked or unexpected_ids:
        raise AssertionError(
            "Identity or draft-outcome fields reached the model table: "
            f"{sorted(set(leaked + unexpected_ids))}"
        )

    cohort["position"] = cohort["position"].astype("string")
    cohort = cohort[
        KEY_COLUMNS + ["position"] + feature_columns + ["drafted"]
    ].sort_values(KEY_COLUMNS).reset_index(drop=True)
    cohort.to_parquet(output_path, index=False)

    class_balance = (
        cohort.groupby("draft_year", as_index=False)
        .agg(rows=("drafted", "size"), drafted=("drafted", "sum"))
        .set_index("draft_year")
        .reindex(range(start_year, end_year + 1), fill_value=0)
        .rename_axis("draft_year")
        .reset_index()
    )
    class_balance["undrafted"] = (
        class_balance["rows"] - class_balance["drafted"]
    )
    class_balance["drafted_rate"] = class_balance["drafted"].div(
        class_balance["rows"].where(class_balance["rows"].ne(0))
    )
    _write_csv(class_balance, audit_dir / "class_balance_by_year.csv")

    source_summary_rows = []
    common_keys = len(cohort)
    for source_name, frame in source_frames.items():
        source_summary_rows.append(
            {
                "source": source_name,
                "unique_player_year_keys": len(frame),
                "numeric_features": sum(
                    column.startswith("combine_") for column in frame.columns
                ),
                "keys_in_four_source_intersection": common_keys,
                "keys_outside_intersection": len(frame) - common_keys,
            }
        )
    source_summary = pd.DataFrame(source_summary_rows)
    _write_csv(source_summary, audit_dir / "combine_source_summary.csv")

    missingness = build_missingness_audit(cohort, feature_columns)
    _write_csv(
        missingness,
        audit_dir / "combine_feature_missingness_by_year.csv",
    )

    dataset_summary = pd.DataFrame(
        [
            {
                "start_year": start_year,
                "end_year": end_year,
                "player_year_rows": len(cohort),
                "unique_player_ids": cohort["nba_player_id"].nunique(),
                "numeric_features": len(feature_columns),
                "drafted": int(cohort["drafted"].sum()),
                "undrafted": int(cohort["drafted"].eq(0).sum()),
                "drafted_rate": cohort["drafted"].mean(),
                "repeated_player_ids": len(repeated_ids),
                "duplicate_source_key_rows": len(duplicate_rows),
                "source_key_mismatches": len(source_key_mismatches),
                "position_conflicts": len(position_conflicts),
            }
        ]
    )
    _write_csv(dataset_summary, audit_dir / "dataset_summary.csv")

    print("\nCombine source summary")
    print(source_summary.to_string(index=False))
    print("\nClass balance by draft year")
    print(class_balance.to_string(index=False))
    print(
        "\nCohort scope: combine participants present in all four source "
        "rosters. This is a post-combine, selection-conditional forecast."
    )
    print(
        "Shooting caveat: roster presence does not guarantee observed shooting "
        "measurements. Nulls were retained and no missingness flags were added."
    )
    print(
        f"Saved {len(cohort):,} rows with {int(cohort['drafted'].sum()):,} "
        f"drafted and {int(cohort['drafted'].eq(0).sum()):,} undrafted to "
        f"{output_path}"
    )
    print(f"Audit CSVs: {audit_dir}")
    return cohort


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the combine cohort builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a post-combine drafted/not-drafted dataset from the exact "
            "intersection of four NBA Draft Combine source rosters."
        ),
        epilog=(
            "Shooting endpoints may have roster rows with null measurements. "
            "The builder preserves nulls and does not add missingness flags."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root containing nba/combine and nba/draft",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Processed-data root used for default output and audits",
    )
    parser.add_argument("--draft-start", type=int, default=2009)
    parser.add_argument("--draft-end", type=int, default=2022)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output Parquet path; defaults to "
            "<processed-root>/draft_classification_dataset.parquet"
        ),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help=(
            "Audit CSV directory; defaults to "
            "<processed-root>/audit/draft_classification"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    build_draft_classification_dataset(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        start_year=args.draft_start,
        end_year=args.draft_end,
        output_path=args.output,
        audit_dir=args.audit_dir,
    )


if __name__ == "__main__":
    main()
