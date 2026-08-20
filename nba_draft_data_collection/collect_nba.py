from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from nba_api.stats.endpoints import (
    draftcombinedrillresults,
    draftcombinenonstationaryshooting,
    draftcombineplayeranthro,
    draftcombinespotshooting,
    drafthistory,
    leaguedashplayerstats,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

LOGGER = logging.getLogger("collect_nba")


def nba_season(start_year: int) -> str:
    """2019 -> '2019-20'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def ensure_dirs(root: Path) -> None:
    for rel in [
        "draft",
        "combine/anthro",
        "combine/drills",
        "combine/spot_shooting",
        "combine/nonstationary_shooting",
        "player_stats/base",
        "player_stats/advanced",
        "raw_json/draft",
        "raw_json/combine/anthro",
        "raw_json/combine/drills",
        "raw_json/combine/spot_shooting",
        "raw_json/combine/nonstationary_shooting",
        "raw_json/player_stats/base",
        "raw_json/player_stats/advanced",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def save_endpoint(endpoint, parquet_path: Path, json_path: Path) -> pd.DataFrame:
    """Persist both a normalized table and the raw NBA response."""
    frames = endpoint.get_data_frames()
    if not frames:
        raise RuntimeError(f"No DataFrame returned for {parquet_path.name}")

    df = frames[0].copy()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    raw = endpoint.get_json()
    with gzip.open(json_path, "wt", encoding="utf-8") as f:
        f.write(raw)

    return df


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=2, max=90),
    reraise=True,
)
def request_with_retry(factory: Callable):
    return factory()


def collect_one(
    factory: Callable,
    parquet_path: Path,
    json_path: Path,
    *,
    force: bool,
    sleep_min: float,
    sleep_max: float,
) -> None:
    if parquet_path.exists() and not force:
        LOGGER.info("SKIP %s", parquet_path)
        return

    LOGGER.info("GET  %s", parquet_path)
    failure_marker = parquet_path.with_suffix(".failed.txt")

    try:
        endpoint = request_with_retry(factory)
        df = save_endpoint(endpoint, parquet_path, json_path)
    except Exception as exc:
        LOGGER.exception("FAILED %s", parquet_path)
        failure_marker.write_text(
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        return

    if failure_marker.exists():
        failure_marker.unlink()

    LOGGER.info("SAVE %s rows -> %s", len(df), parquet_path)
    time.sleep(random.uniform(sleep_min, sleep_max))


def collect_draft(
    root: Path,
    years: Iterable[int],
    *,
    force: bool,
    sleep_min: float,
    sleep_max: float,
) -> None:
    for year in years:
        parquet_path = root / "draft" / f"draft_{year}.parquet"
        json_path = root / "raw_json" / "draft" / f"draft_{year}.json.gz"

        collect_one(
            lambda year=year: drafthistory.DraftHistory(
                league_id="00",
                season_year_nullable=str(year),
                timeout=60,
            ),
            parquet_path,
            json_path,
            force=force,
            sleep_min=sleep_min,
            sleep_max=sleep_max,
        )


def collect_combine(
    root: Path,
    years: Iterable[int],
    *,
    force: bool,
    sleep_min: float,
    sleep_max: float,
) -> None:
    # These endpoints take the draft/combine year directly, e.g. SeasonYear=2019.
    endpoints = [
        (
            "anthro",
            draftcombineplayeranthro.DraftCombinePlayerAnthro,
        ),
        (
            "drills",
            draftcombinedrillresults.DraftCombineDrillResults,
        ),
        (
            "spot_shooting",
            draftcombinespotshooting.DraftCombineSpotShooting,
        ),
        (
            "nonstationary_shooting",
            draftcombinenonstationaryshooting.DraftCombineNonStationaryShooting,
        ),
    ]

    for year in years:
        for name, cls in endpoints:
            parquet_path = root / "combine" / name / f"{name}_{year}.parquet"
            json_path = (
                root / "raw_json" / "combine" / name / f"{name}_{year}.json.gz"
            )

            collect_one(
                lambda year=year, cls=cls: cls(
                    league_id="00",
                    season_year=str(year),
                    timeout=60,
                ),
                parquet_path,
                json_path,
                force=force,
                sleep_min=sleep_min,
                sleep_max=sleep_max,
            )


def collect_player_stats(
    root: Path,
    season_start_years: Iterable[int],
    *,
    force: bool,
    sleep_min: float,
    sleep_max: float,
) -> None:
    measures = {
        "base": "Base",
        "advanced": "Advanced",
    }

    for start_year in season_start_years:
        season = nba_season(start_year)

        for folder, measure in measures.items():
            parquet_path = (
                root / "player_stats" / folder / f"{folder}_{season}.parquet"
            )
            json_path = (
                root
                / "raw_json"
                / "player_stats"
                / folder
                / f"{folder}_{season}.json.gz"
            )

            collect_one(
                lambda season=season, measure=measure: (
                    leaguedashplayerstats.LeagueDashPlayerStats(
                        season=season,
                        season_type_all_star="Regular Season",
                        measure_type_detailed_defense=measure,
                        per_mode_detailed="Totals",
                        timeout=60,
                    )
                ),
                parquet_path,
                json_path,
                force=force,
                sleep_min=sleep_min,
                sleep_max=sleep_max,
            )


def combine_draft_files(root: Path) -> None:
    files = sorted((root / "draft").glob("draft_*.parquet"))
    if not files:
        return

    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df = df.drop_duplicates()
    out = root / "draft" / "draft_all.parquet"
    df.to_parquet(out, index=False)
    LOGGER.info("SAVE %s rows -> %s", len(df), out)


def combine_nba_stats(root: Path, folder: str) -> None:
    files = sorted((root / "player_stats" / folder).glob(f"{folder}_*.parquet"))
    if not files:
        return

    chunks = []
    for p in files:
        if p.name.endswith("_all.parquet"):
            continue
        df = pd.read_parquet(p).copy()
        season = p.stem.removeprefix(f"{folder}_")
        df["SEASON"] = season
        chunks.append(df)

    if not chunks:
        return

    out_df = pd.concat(chunks, ignore_index=True)
    out = root / "player_stats" / folder / f"{folder}_all.parquet"
    out_df.to_parquet(out, index=False)
    LOGGER.info("SAVE %s rows -> %s", len(out_df), out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect NBA draft, combine and season player statistics."
    )
    p.add_argument("--draft-start", type=int, default=2009)
    p.add_argument("--draft-end", type=int, default=2022)
    p.add_argument(
        "--nba-season-start",
        type=int,
        default=None,
        help="Starting year of NBA season. Default: draft-start.",
    )
    p.add_argument(
        "--nba-season-end",
        type=int,
        default=None,
        help=(
            "Starting year of final NBA season. "
            "Default: draft-end + 2, enough for 3 post-draft seasons."
        ),
    )
    p.add_argument("--output", type=Path, default=Path("data/raw/nba"))
    p.add_argument("--sleep-min", type=float, default=2.0)
    p.add_argument("--sleep-max", type=float, default=4.0)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--skip-combine-shooting",
        action="store_true",
        help="Only collect combine anthropometrics and drills.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.draft_end < args.draft_start:
        raise ValueError("draft-end must be >= draft-start")

    season_start = (
        args.nba_season_start
        if args.nba_season_start is not None
        else args.draft_start
    )
    season_end = (
        args.nba_season_end
        if args.nba_season_end is not None
        else args.draft_end + 2
    )

    root = args.output
    ensure_dirs(root)

    draft_years = range(args.draft_start, args.draft_end + 1)
    nba_seasons = range(season_start, season_end + 1)

    collect_draft(
        root,
        draft_years,
        force=args.force,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
    )

    if args.skip_combine_shooting:
        # Same implementation but only anthro + drills.
        for year in draft_years:
            for name, cls in [
                ("anthro", draftcombineplayeranthro.DraftCombinePlayerAnthro),
                ("drills", draftcombinedrillresults.DraftCombineDrillResults),
            ]:
                collect_one(
                    lambda year=year, cls=cls: cls(
                        league_id="00",
                        season_year=str(year),
                        timeout=60,
                    ),
                    root / "combine" / name / f"{name}_{year}.parquet",
                    root
                    / "raw_json"
                    / "combine"
                    / name
                    / f"{name}_{year}.json.gz",
                    force=args.force,
                    sleep_min=args.sleep_min,
                    sleep_max=args.sleep_max,
                )
    else:
        collect_combine(
            root,
            draft_years,
            force=args.force,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
        )

    collect_player_stats(
        root,
        nba_seasons,
        force=args.force,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
    )

    combine_draft_files(root)
    combine_nba_stats(root, "base")
    combine_nba_stats(root, "advanced")

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
