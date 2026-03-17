from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_sharded_results(*, shard_root: Path, out_csv: Path) -> Path:
    shard_paths = sorted(shard_root.glob("shard_*.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard CSVs found in {shard_root}")

    frames = []
    for path in shard_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_shard_csv"] = str(path)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"Shard CSVs were present but empty under {shard_root}")

    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ("dataset", "row_index") if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).drop_duplicates(sort_cols, keep="last")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[MMGBSA][MERGE] shards={len(shard_paths)} rows={len(merged)} out={out_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded MMGBSA result CSVs.")
    parser.add_argument("--shard-root", required=True, help="Directory containing shard_*.csv outputs")
    parser.add_argument("--out", required=True, help="Merged output CSV path")
    args = parser.parse_args()
    merge_sharded_results(shard_root=Path(args.shard_root), out_csv=Path(args.out))


if __name__ == "__main__":
    main()
