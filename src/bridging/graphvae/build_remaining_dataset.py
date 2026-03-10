from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pandas as pd
import torch

from bridging.utils.dataset_rows import row_pdb_id

from .ids import canonical_complex_id, sanitize_filename_token


def _load_done_complex_ids(checkpoint_dir: Path) -> set[str]:
    done: set[str] = set()
    if not checkpoint_dir.exists():
        return done
    for shard in sorted(checkpoint_dir.glob("records_*.pt")):
        chunk = torch.load(shard, map_location="cpu")
        if not isinstance(chunk, list):
            raise RuntimeError(f"Invalid checkpoint shard format: {shard}")
        for rec in chunk:
            cid = str(rec.get("complex_id", "")).strip()
            if cid:
                done.add(cid)
    return done


def _load_hdf5_model_ids(hdf5_paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in hdf5_paths:
        with h5py.File(path, "r") as h5:
            for key in h5.keys():
                out.add(str(key).split(":")[-1].strip())
    return out


def build_remaining_dataset(
    *,
    dataset_csv: Path,
    md_root: Path,
    hdf5_paths: list[Path],
    checkpoint_dir: Path,
    out_csv: Path,
    require_done: bool,
    require_graph: bool,
) -> dict:
    df = pd.read_csv(dataset_csv)
    done_complex_ids = _load_done_complex_ids(checkpoint_dir)
    graph_model_ids = _load_hdf5_model_ids(hdf5_paths) if require_graph else set()

    selected_rows = []
    skipped_missing_id = 0
    skipped_done = 0
    skipped_not_done = 0
    skipped_no_graph = 0

    for row in df.to_dict("records"):
        complex_id = canonical_complex_id(row)
        if complex_id is None:
            skipped_missing_id += 1
            continue
        if complex_id in done_complex_ids:
            skipped_done += 1
            continue

        pdb_id = row_pdb_id(row)
        if pdb_id is None:
            skipped_missing_id += 1
            continue

        if require_done and not (md_root / str(pdb_id) / "DONE").exists():
            skipped_not_done += 1
            continue

        if require_graph and sanitize_filename_token(complex_id) not in graph_model_ids:
            skipped_no_graph += 1
            continue

        selected_rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(out_csv, index=False)
    else:
        df.head(0).to_csv(out_csv, index=False)

    report = {
        "dataset_csv": str(dataset_csv),
        "md_root": str(md_root),
        "checkpoint_dir": str(checkpoint_dir),
        "out_csv": str(out_csv),
        "rows_input": int(len(df)),
        "rows_output": int(len(selected_rows)),
        "done_checkpointed": int(len(done_complex_ids)),
        "graph_models": int(len(graph_model_ids)),
        "skipped_missing_id": int(skipped_missing_id),
        "skipped_done": int(skipped_done),
        "skipped_not_done": int(skipped_not_done),
        "skipped_no_graph": int(skipped_no_graph),
        "require_done": bool(require_done),
        "require_graph": bool(require_graph),
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 'remaining rows' CSV by removing complexes already present in prepare "
            "checkpoint shards and optionally requiring DONE+DeepRank overlap."
        )
    )
    parser.add_argument("--dataset", required=True, help="Input dataset CSV.")
    parser.add_argument("--md-root", required=True, help="MD root containing <pdb_id>/DONE.")
    parser.add_argument(
        "--deep-rank-hdf5",
        nargs="*",
        default=[],
        help="DeepRank HDF5 path(s). Required when --require-graph is set.",
    )
    parser.add_argument("--checkpoint-dir", required=True, help="Existing prepared/checkpoints directory.")
    parser.add_argument("--out-csv", required=True, help="Output CSV for remaining rows.")
    parser.add_argument("--require-done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-graph", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    hdf5_paths = [Path(p) for p in args.deep_rank_hdf5]
    if args.require_graph and not hdf5_paths:
        raise ValueError("Provide --deep-rank-hdf5 when --require-graph is enabled.")
    report = build_remaining_dataset(
        dataset_csv=Path(args.dataset),
        md_root=Path(args.md_root),
        hdf5_paths=hdf5_paths,
        checkpoint_dir=Path(args.checkpoint_dir),
        out_csv=Path(args.out_csv),
        require_done=bool(args.require_done),
        require_graph=bool(args.require_graph),
    )
    print(
        "[REMAIN] "
        f"input={report['rows_input']} output={report['rows_output']} "
        f"done_checkpointed={report['done_checkpointed']} "
        f"skipped_done={report['skipped_done']} "
        f"skipped_not_done={report['skipped_not_done']} "
        f"skipped_no_graph={report['skipped_no_graph']}"
    )
    print(f"[REMAIN] out_csv={report['out_csv']}")


if __name__ == "__main__":
    main()
