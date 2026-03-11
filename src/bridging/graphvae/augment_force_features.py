from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bridging.MD.paths import PDB_CACHE_DIR
from bridging.MD.prefetch_pdbs import ensure_pdb_cached
from bridging.utils.dataset_rows import row_chain_groups, row_pdb_id

from .config import FORCE_NODE_INPUT_FEATURES
from .force_features import compute_node_interchain_force_features
from .ids import canonical_complex_id
from .md_dynamics import load_protein_md_trajectory


def _sorted_shards(checkpoint_dir: Path) -> list[Path]:
    return sorted(p for p in checkpoint_dir.glob("records_*.pt") if p.is_file())


def _fmt_pct(num: int, den: int) -> float:
    return 100.0 * float(num) / float(max(1, den))


def _apply_columns(
    arr: np.ndarray,
    names: list[str],
    add_arr: np.ndarray,
    add_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    out = np.asarray(arr, dtype=np.float32)
    out_names = list(names)
    for j, name in enumerate(add_names):
        col = np.asarray(add_arr[:, j], dtype=np.float32).reshape(-1, 1)
        if name in out_names:
            idx = out_names.index(name)
            out[:, idx : idx + 1] = col
        else:
            out = np.concatenate([out, col], axis=1)
            out_names.append(name)
    return out, out_names


def _dataset_lookup(dataset_path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    df = pd.read_csv(dataset_path)
    for row in df.to_dict("records"):
        complex_id = canonical_complex_id(row)
        chains_1, chains_2 = row_chain_groups(row)
        pdb_id = row_pdb_id(row)
        if complex_id is None or chains_1 is None or chains_2 is None or not pdb_id:
            continue
        if complex_id in lookup:
            continue
        lookup[complex_id] = {
            "pdb_id": str(pdb_id).strip(),
            "ligand_group": str(chains_1).strip(),
            "receptor_group": str(chains_2).strip(),
        }
    return lookup


def _augment_records_list(
    *,
    records: list[dict],
    dataset_lookup: dict[str, dict],
    md_root: Path,
    pdb_cache_root: Path,
    max_frames: int,
    traj_cache_size: int,
    progress_every: int,
) -> tuple[list[dict], dict]:
    traj_cache: OrderedDict[str, object] = OrderedDict()
    cache_lim = max(0, int(traj_cache_size))
    out_records: list[dict] = []

    totals = {
        "n_records": int(len(records)),
        "n_augmented": 0,
        "n_failed": 0,
        "n_nodes": 0,
        "n_mapped_nodes": 0,
        "n_frames_total": 0,
        "fail_examples": [],
        "source_mode": "protein_traj_force_analysis",
    }

    for i, rec in enumerate(records, start=1):
        rec_out = dict(rec)
        complex_id = str(rec.get("complex_id", "")).strip()
        meta = dataset_lookup.get(complex_id)
        if meta is None:
            totals["n_failed"] += 1
            if len(totals["fail_examples"]) < 20:
                totals["fail_examples"].append(
                    {
                        "complex_id": complex_id,
                        "pdb_id": str(rec.get("pdb_id", "")).strip(),
                        "error": "missing_dataset_lookup",
                    }
                )
            out_records.append(rec_out)
            continue

        pdb_id = str(meta["pdb_id"]).strip()
        md_dir = md_root / pdb_id
        raw_pdb_path, _ = ensure_pdb_cached(pdb_id, cache_dir=pdb_cache_root)
        protein_topology_pdb = md_dir / "topology_protein.pdb"

        try:
            if pdb_id in traj_cache:
                traj = traj_cache.pop(pdb_id)
                traj_cache[pdb_id] = traj
            else:
                traj = load_protein_md_trajectory(md_dir, max_frames=max_frames)
                if cache_lim > 0:
                    traj_cache[pdb_id] = traj
                    if len(traj_cache) > cache_lim:
                        traj_cache.popitem(last=False)

            force_arr, stats = compute_node_interchain_force_features(
                traj=traj,
                topology_pdb=protein_topology_pdb,
                raw_pdb_path=raw_pdb_path,
                ligand_group=str(meta["ligand_group"]),
                receptor_group=str(meta["receptor_group"]),
                node_chain_id=list(rec["node_chain_id"]),
                node_position=[int(x) for x in rec["node_position"]],
            )
            node = rec["node_features"]
            node_arr = node.detach().cpu().numpy() if torch.is_tensor(node) else np.asarray(node)
            node_names = list(rec["node_feature_names"])
            node_aug, node_names_aug = _apply_columns(
                node_arr,
                node_names,
                force_arr,
                list(FORCE_NODE_INPUT_FEATURES),
            )
            rec_out["node_features"] = torch.as_tensor(node_aug, dtype=torch.float32)
            rec_out["node_feature_names"] = node_names_aug
            totals["n_augmented"] += 1
            totals["n_nodes"] += int(stats["n_nodes"])
            totals["n_mapped_nodes"] += int(stats["n_mapped_nodes"])
            totals["n_frames_total"] += int(stats["n_frames"])
        except Exception as exc:  # pragma: no cover - cluster/runtime dependent
            totals["n_failed"] += 1
            if len(totals["fail_examples"]) < 20:
                totals["fail_examples"].append(
                    {
                        "complex_id": complex_id,
                        "pdb_id": pdb_id,
                        "error": repr(exc),
                    }
                )
        out_records.append(rec_out)

        if progress_every > 0 and (i % progress_every == 0 or i == len(records)):
            print(
                f"[FORCE] progress {i}/{len(records)} "
                f"augmented={totals['n_augmented']} failed={totals['n_failed']} "
                f"mapped_nodes={totals['n_mapped_nodes']}/{max(1, totals['n_nodes'])}",
                flush=True,
            )

    totals["mapping_coverage_pct"] = _fmt_pct(totals["n_mapped_nodes"], totals["n_nodes"])
    totals["mean_frames_per_record"] = (
        float(totals["n_frames_total"]) / float(max(1, totals["n_augmented"]))
    )
    return out_records, totals


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append node-level inter-chain force features to prepared graph records.")
    p.add_argument("--dataset", required=True, help="Dataset CSV used to recover ligand/receptor chain groups.")
    p.add_argument("--md-root", required=True, help="Root with per-PDB MD outputs.")
    p.add_argument("--pdb-cache-root", default=str(PDB_CACHE_DIR))
    p.add_argument("--records-in", help="Input graph_records.pt")
    p.add_argument("--records-out", help="Output graph_records.pt")
    p.add_argument("--checkpoint-dir-in", help="Input checkpoint dir with records_*.pt")
    p.add_argument("--checkpoint-dir-out", help="Output checkpoint dir for augmented shards")
    p.add_argument("--report-out", required=True, help="Output JSON report path")
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--traj-cache-size", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_lookup = _dataset_lookup(Path(args.dataset))
    md_root = Path(args.md_root).expanduser()
    pdb_cache_root = Path(args.pdb_cache_root).expanduser()
    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    if bool(args.records_in) == bool(args.checkpoint_dir_in):
        raise ValueError("Provide exactly one of --records-in or --checkpoint-dir-in.")

    aggregate = {
        "mode": "records" if args.records_in else "checkpoints",
        "dataset": str(Path(args.dataset)),
        "md_root": str(md_root),
        "pdb_cache_root": str(pdb_cache_root),
        "force_features": list(FORCE_NODE_INPUT_FEATURES),
        "parts": [],
    }

    if args.records_in:
        if not args.records_out:
            raise ValueError("--records-out is required with --records-in")
        records_in = Path(args.records_in)
        records_out = Path(args.records_out)
        records = torch.load(records_in, map_location="cpu")
        if not isinstance(records, list):
            raise RuntimeError(f"Invalid records file: {records_in}")
        out, stats = _augment_records_list(
            records=records,
            dataset_lookup=dataset_lookup,
            md_root=md_root,
            pdb_cache_root=pdb_cache_root,
            max_frames=int(args.max_frames),
            traj_cache_size=int(args.traj_cache_size),
            progress_every=int(args.progress_every),
        )
        if int(stats["n_failed"]) > 0:
            raise RuntimeError(
                f"Force feature augmentation failed for {stats['n_failed']}/{stats['n_records']} records "
                f"in {records_in}. Inspect the report and logs before merging."
            )
        records_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, records_out)
        aggregate["parts"].append(
            {
                "input": str(records_in),
                "output": str(records_out),
                **stats,
            }
        )
        print(
            f"[FORCE] records augmented={stats['n_augmented']}/{stats['n_records']} "
            f"mapping={stats['mapping_coverage_pct']:.2f}% frames_per_record={stats['mean_frames_per_record']:.1f}",
            flush=True,
        )
    else:
        if not args.checkpoint_dir_out:
            raise ValueError("--checkpoint-dir-out is required with --checkpoint-dir-in")
        ck_in = Path(args.checkpoint_dir_in)
        ck_out = Path(args.checkpoint_dir_out)
        ck_out.mkdir(parents=True, exist_ok=True)
        shards = _sorted_shards(ck_in)
        if not shards:
            raise RuntimeError(f"No checkpoint shards found in {ck_in}")
        for shard in shards:
            records = torch.load(shard, map_location="cpu")
            if not isinstance(records, list):
                raise RuntimeError(f"Invalid shard format: {shard}")
            out, stats = _augment_records_list(
                records=records,
                dataset_lookup=dataset_lookup,
                md_root=md_root,
                pdb_cache_root=pdb_cache_root,
                max_frames=int(args.max_frames),
                traj_cache_size=int(args.traj_cache_size),
                progress_every=int(args.progress_every),
            )
            if int(stats["n_failed"]) > 0:
                raise RuntimeError(
                    f"Force feature augmentation failed for {stats['n_failed']}/{stats['n_records']} records "
                    f"in {shard}. Inspect the logs before merging."
                )
            out_path = ck_out / shard.name
            torch.save(out, out_path)
            aggregate["parts"].append(
                {
                    "input": str(shard),
                    "output": str(out_path),
                    **stats,
                }
            )
            print(
                f"[FORCE] shard={shard.name} augmented={stats['n_augmented']}/{stats['n_records']} "
                f"mapping={stats['mapping_coverage_pct']:.2f}%",
                flush=True,
            )

    n_records = sum(int(p.get("n_records", 0)) for p in aggregate["parts"])
    n_augmented = sum(int(p.get("n_augmented", 0)) for p in aggregate["parts"])
    n_failed = sum(int(p.get("n_failed", 0)) for p in aggregate["parts"])
    n_nodes = sum(int(p.get("n_nodes", 0)) for p in aggregate["parts"])
    n_mapped = sum(int(p.get("n_mapped_nodes", 0)) for p in aggregate["parts"])
    frames_total = sum(int(p.get("n_frames_total", 0)) for p in aggregate["parts"])
    aggregate["summary"] = {
        "n_parts": int(len(aggregate["parts"])),
        "n_records": int(n_records),
        "n_augmented": int(n_augmented),
        "n_failed": int(n_failed),
        "n_nodes": int(n_nodes),
        "n_mapped_nodes": int(n_mapped),
        "mapping_coverage_pct": _fmt_pct(n_mapped, n_nodes),
        "mean_frames_per_record": float(frames_total) / float(max(1, n_augmented)),
    }
    report_out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"[FORCE] report={report_out}", flush=True)
    print(
        f"[FORCE] summary records={n_augmented}/{n_records} "
        f"mapping={aggregate['summary']['mapping_coverage_pct']:.2f}% "
        f"frames_per_record={aggregate['summary']['mean_frames_per_record']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
