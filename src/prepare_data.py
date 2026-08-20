"""Create the small table shared by the portfolio models."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROSPECTS = ROOT / "nba_draft_data_collection/data/processed/draft_classification_advanced.parquet"
OUTCOMES = ROOT / "nba_draft_data_collection/data/processed/drafted_players_advanced.parquet"
OUTPUT = ROOT / "nba_draft_data_collection/data/processed/two_stage_college_comparison.parquet"


def main() -> None:
    prospects = pd.read_parquet(PROSPECTS)
    outcomes = pd.read_parquet(
        OUTCOMES, columns=["nba_player_id", "draft_year", "nba_min_3y"]
    )
    data = prospects.merge(
        outcomes,
        on=["nba_player_id", "draft_year"],
        how="left",
        validate="one_to_one",
    )
    if data.loc[data["drafted"].eq(0), "nba_min_3y"].notna().any():
        raise ValueError("An undrafted player was given a drafted-player outcome")
    data["college_position"] = data["college_position"].fillna("missing")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT, index=False)
    print(f"Saved {len(data)} prospect-years to {OUTPUT}")


if __name__ == "__main__":
    main()
