from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _pdb_col(fieldnames: list[str]) -> str | None:
    for col in ("PDB", "pdb", "PDB_ID", "pdb_id"):
        if col in fieldnames:
            return col
    return None


def _load_rows(dataset: Path) -> tuple[list[str], list[dict[str, str]]]:
    with dataset.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"No CSV header found: {dataset}")
    return fieldnames, rows


def _dedup_rows_by_pdb(rows: list[dict[str, str]], pdb_col: str | None) -> tuple[list[dict[str, str]], int]:
    if pdb_col is None:
        return rows, 0

    kept: list[dict[str, str]] = []
    seen = set()
    dropped = 0
    for row in rows:
        pdb = str(row.get(pdb_col, "")).strip().upper()
        row[pdb_col] = pdb
        if pdb in seen:
            dropped += 1
            continue
        seen.add(pdb)
        kept.append(row)
    return kept, dropped


def _write_shard(out_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def preshard_dataset(dataset: str | Path, shard_dir: str | Path | None = None, num_shards: int = 2) -> Path:
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if num_shards < 1:
        raise ValueError(f"num-shards must be >= 1, got {num_shards}")

    if shard_dir is None:
        shard_root = dataset_path.parent / "shards" / dataset_path.stem
    else:
        shard_root = Path(shard_dir)

    fieldnames, rows = _load_rows(dataset_path)
    pdb_col = _pdb_col(fieldnames)
    rows, dropped_dupes = _dedup_rows_by_pdb(rows, pdb_col)

    shards: list[list[dict[str, str]]] = [[] for _ in range(num_shards)]
    for idx, row in enumerate(rows):
        shards[idx % num_shards].append(row)

    for shard_id, shard_rows in enumerate(shards):
        out_path = shard_root / f"shard_{shard_id:02d}.csv"
        _write_shard(out_path, fieldnames, shard_rows)
        print(f"[SHARD] id={shard_id} rows={len(shard_rows)} out={out_path}")

    print(
        f"[DONE] dataset={dataset_path} total_rows={len(rows)} "
        f"dropped_pdb_dupes={dropped_dupes} num_shards={num_shards} shard_dir={shard_root}"
    )
    return shard_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute disjoint dataset shards into CSV files.")
    parser.add_argument("--dataset", required=True, help="Input dataset CSV")
    parser.add_argument("--shard-dir", help="Output shard directory (default: processedData/shards/<dataset_stem>)")
    parser.add_argument("--num-shards", type=int, default=2, help="Number of shards (default: 2)")
    args = parser.parse_args()

    preshard_dataset(
        dataset=args.dataset,
        shard_dir=args.shard_dir,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
