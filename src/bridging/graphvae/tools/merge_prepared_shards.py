from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..common.splits import make_train_val_test_split


def _iter_shards(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []
    return sorted(p for p in checkpoint_dir.glob("records_*.pt") if p.is_file())


def _load_records_from_shards(shards: list[Path]) -> list[dict]:
    records: list[dict] = []
    for shard in shards:
        chunk = torch.load(shard, map_location="cpu")
        if not isinstance(chunk, list):
            raise RuntimeError(f"Invalid checkpoint shard format: {shard}")
        records.extend(chunk)
    return records


def merge_prepared_shards(
    *,
    base_checkpoint_dir: Path | None,
    shard_root: Path,
    out_dir: Path,
    train_fraction: float,
    val_fraction: float,
    split_seed: int,
    dataset_csv: Path | None,
) -> dict:
    shard_checkpoint_dirs = sorted(p for p in shard_root.glob("shard_*/prepared/checkpoints") if p.is_dir())
    shard_paths: list[Path] = []
    for d in shard_checkpoint_dirs:
        shard_paths.extend(_iter_shards(d))

    base_paths: list[Path] = []
    if base_checkpoint_dir is not None:
        base_paths = _iter_shards(base_checkpoint_dir)

    all_paths = [*base_paths, *shard_paths]
    if not all_paths:
        raise RuntimeError(
            "No checkpoint shards found to merge. "
            f"base_checkpoint_dir={base_checkpoint_dir} shard_root={shard_root}"
        )

    dedup: dict[str, dict] = {}
    n_loaded = 0
    for rec in _load_records_from_shards(all_paths):
        n_loaded += 1
        cid = str(rec.get("complex_id", "")).strip()
        if not cid:
            continue
        dedup[cid] = rec

    records = list(dedup.values())
    split_map = make_train_val_test_split(
        [r["complex_id"] for r in records],
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=split_seed,
    )
    for rec in records:
        rec["split"] = split_map[rec["complex_id"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "graph_records.pt"
    splits_path = out_dir / "splits.json"
    report_path = out_dir / "prepare_report.json"

    torch.save(records, records_path)
    splits_path.write_text(json.dumps(split_map, indent=2), encoding="utf-8")

    split_counts = {
        "train": int(sum(1 for r in records if r["split"] == "train")),
        "val": int(sum(1 for r in records if r["split"] == "val")),
        "test": int(sum(1 for r in records if r["split"] == "test")),
    }
    report = {
        "dataset_csv": str(dataset_csv) if dataset_csv is not None else None,
        "records_path": str(records_path),
        "splits_path": str(splits_path),
        "base_checkpoint_dir": str(base_checkpoint_dir) if base_checkpoint_dir is not None else None,
        "shard_root": str(shard_root),
        "n_base_shards": int(len(base_paths)),
        "n_new_shards": int(len(shard_paths)),
        "n_loaded_records": int(n_loaded),
        "n_records": int(len(records)),
        "split_seed": int(split_seed),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "split_counts": split_counts,
        "note": "Merged from checkpoint shards and re-split globally.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if len(records) < 1:
        raise RuntimeError("Merged records are empty.")
    if min(split_counts.values()) < 1:
        raise RuntimeError(
            "Merged records produced empty split. "
            f"split_counts={split_counts}. Adjust split seed/fractions."
        )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge base + sharded prepare checkpoint outputs into a single prepared dataset."
    )
    parser.add_argument("--base-checkpoint-dir", help="Optional existing prepared/checkpoints dir.")
    parser.add_argument("--shard-root", required=True, help="Root containing shard_*/prepared/checkpoints.")
    parser.add_argument("--out-dir", required=True, help="Output prepared dir for merged graph_records.pt.")
    parser.add_argument("--dataset", help="Optional dataset CSV path for report metadata.")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = merge_prepared_shards(
        base_checkpoint_dir=Path(args.base_checkpoint_dir) if args.base_checkpoint_dir else None,
        shard_root=Path(args.shard_root),
        out_dir=Path(args.out_dir),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        dataset_csv=Path(args.dataset) if args.dataset else None,
    )
    print(
        "[MERGE] "
        f"records={report['n_records']} "
        f"base_shards={report['n_base_shards']} "
        f"new_shards={report['n_new_shards']} "
        f"split_counts={report['split_counts']}"
    )
    print(f"[MERGE] records_path={report['records_path']}")


if __name__ == "__main__":
    main()
