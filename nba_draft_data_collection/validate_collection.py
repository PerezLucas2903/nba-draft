from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(args.root.rglob("*.parquet"))

    if not files:
        print(f"No parquet files found under {args.root}")
        return

    errors = []
    total_rows = 0

    for path in files:
        try:
            df = pd.read_parquet(path)
            rows, cols = df.shape
            total_rows += rows
            print(f"{rows:>9,} rows | {cols:>3} cols | {path}")
        except Exception as exc:
            errors.append((path, exc))
            print(f"ERROR | {path} | {exc}")

    print()
    print(f"Files: {len(files)}")
    print(f"Rows across files (not deduplicated): {total_rows:,}")
    print(f"Read errors: {len(errors)}")


if __name__ == "__main__":
    main()
