from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NCAA + NBA collection.")
    p.add_argument("--draft-start", type=int, default=2009)
    p.add_argument("--draft-end", type=int, default=2022)
    p.add_argument("--data-root", type=Path, default=Path("data/raw"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript was not found in PATH. Install R or run collect_nba.py "
            "separately and execute collect_college.R from an R environment."
        )

    college_root = args.data_root / "college"
    nba_root = args.data_root / "nba"

    run(
        [
            rscript,
            "collect_college.R",
            str(args.draft_start),
            str(args.draft_end),
            str(college_root),
            "true" if args.force else "false",
        ]
    )

    run(
        [
            rscript,
            "collect_crosswalks.R",
            str(args.draft_start),
            str(args.draft_end),
            str(args.draft_end + 2),
            str(args.data_root / "crosswalks"),
            "true" if args.force else "false",
        ]
    )

    nba_cmd = [
        sys.executable,
        "collect_nba.py",
        "--draft-start",
        str(args.draft_start),
        "--draft-end",
        str(args.draft_end),
        "--output",
        str(nba_root),
    ]
    if args.force:
        nba_cmd.append("--force")

    run(nba_cmd)


if __name__ == "__main__":
    main()
