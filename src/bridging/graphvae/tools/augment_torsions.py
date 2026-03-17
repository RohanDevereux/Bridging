from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from ..common.config import TORSION_NODE_INPUT_FEATURES
from ..prep.md_dynamics import compute_node_torsion_sincos_features, load_full_md_trajectory


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


def _augment_records_list(
    *,
    records: list[dict],
    md_root: Path,
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
        "n_phi_defined": 0,
        "n_psi_defined": 0,
        "n_ambiguous_keys": 0,
        "fail_examples": [],
    }

    for i, rec in enumerate(records, start=1):
        rec_out = dict(rec)
        pdb_id = str(rec.get("pdb_id", "")).strip()
        md_dir = md_root / pdb_id

        try:
            if pdb_id in traj_cache:
                traj = traj_cache.pop(pdb_id)
                traj_cache[pdb_id] = traj
            else:
                traj = load_full_md_trajectory(md_dir, max_frames=max_frames)
                if cache_lim > 0:
                    traj_cache[pdb_id] = traj
                    if len(traj_cache) > cache_lim:
                        traj_cache.popitem(last=False)

            torsion, stats = compute_node_torsion_sincos_features(
                traj=traj,
                node_chain_id=list(rec["node_chain_id"]),
                node_position=[int(x) for x in rec["node_position"]],
            )
            node = rec["node_features"]
            node_arr = node.detach().cpu().numpy() if torch.is_tensor(node) else np.asarray(node)
            node_names = list(rec["node_feature_names"])
            node_aug, node_names_aug = _apply_columns(
                node_arr,
                node_names,
                torsion,
                list(TORSION_NODE_INPUT_FEATURES),
            )
            rec_out["node_features"] = torch.as_tensor(node_aug, dtype=torch.float32)
            rec_out["node_feature_names"] = node_names_aug
            totals["n_augmented"] += 1
            totals["n_nodes"] += int(stats["n_nodes"])
            totals["n_mapped_nodes"] += int(stats["n_mapped_nodes"])
            totals["n_phi_defined"] += int(stats["n_phi_defined"])
            totals["n_psi_defined"] += int(stats["n_psi_defined"])
            totals["n_ambiguous_keys"] += int(stats["n_ambiguous_keys"])
        except Exception as exc:  # pragma: no cover - defensive for varied cluster data
            totals["n_failed"] += 1
            if len(totals["fail_examples"]) < 20:
                totals["fail_examples"].append(
                    {
                        "complex_id": str(rec.get("complex_id", "")),
                        "pdb_id": pdb_id,
                        "error": repr(exc),
                    }
                )
        out_records.append(rec_out)

        if progress_every > 0 and (i % progress_every == 0 or i == len(records)):
            print(
                f"[TORSION] progress {i}/{len(records)} "
                f"augmented={totals['n_augmented']} failed={totals['n_failed']} "
                f"mapped_nodes={totals['n_mapped_nodes']}/{max(1, totals['n_nodes'])}",
                flush=True,
            )

    totals["mapping_coverage_pct"] = _fmt_pct(totals["n_mapped_nodes"], totals["n_nodes"])
    totals["phi_defined_pct_of_mapped"] = _fmt_pct(totals["n_phi_defined"], totals["n_mapped_nodes"])
    totals["psi_defined_pct_of_mapped"] = _fmt_pct(totals["n_psi_defined"], totals["n_mapped_nodes"])
    return out_records, totals


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append node-level torsion sin/cos features to prepared graph records.")
    p.add_argument("--md-root", required=True, help="Root with per-PDB MD outputs.")
    p.add_argument("--records-in", help="Input graph_records.pt")
    p.add_argument("--records-out", help="Output graph_records.pt")
    p.add_argument("--checkpoint-dir-in", help="Input checkpoint dir with records_*.pt")
    p.add_argument("--checkpoint-dir-out", help="Output checkpoint dir for augmented shards")
    p.add_argument("--report-out", required=True, help="Output JSON report path")
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--traj-cache-size", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    md_root = Path(args.md_root).expanduser()
    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    if bool(args.records_in) == bool(args.checkpoint_dir_in):
        raise ValueError("Provide exactly one of --records-in or --checkpoint-dir-in.")

    aggregate = {
        "mode": "records" if args.records_in else "checkpoints",
        "md_root": str(md_root),
        "torsion_features": list(TORSION_NODE_INPUT_FEATURES),
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
            md_root=md_root,
            max_frames=int(args.max_frames),
            traj_cache_size=int(args.traj_cache_size),
            progress_every=int(args.progress_every),
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
            f"[TORSION] records augmented={stats['n_augmented']}/{stats['n_records']} "
            f"mapping={stats['mapping_coverage_pct']:.2f}% "
            f"phi={stats['phi_defined_pct_of_mapped']:.2f}% psi={stats['psi_defined_pct_of_mapped']:.2f}%",
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
                md_root=md_root,
                max_frames=int(args.max_frames),
                traj_cache_size=int(args.traj_cache_size),
                progress_every=int(args.progress_every),
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
                f"[TORSION] shard={shard.name} augmented={stats['n_augmented']}/{stats['n_records']} "
                f"mapping={stats['mapping_coverage_pct']:.2f}%",
                flush=True,
            )

    n_records = sum(int(p.get("n_records", 0)) for p in aggregate["parts"])
    n_augmented = sum(int(p.get("n_augmented", 0)) for p in aggregate["parts"])
    n_nodes = sum(int(p.get("n_nodes", 0)) for p in aggregate["parts"])
    n_mapped = sum(int(p.get("n_mapped_nodes", 0)) for p in aggregate["parts"])
    n_phi = sum(int(p.get("n_phi_defined", 0)) for p in aggregate["parts"])
    n_psi = sum(int(p.get("n_psi_defined", 0)) for p in aggregate["parts"])
    n_amb = sum(int(p.get("n_ambiguous_keys", 0)) for p in aggregate["parts"])
    n_failed = sum(int(p.get("n_failed", 0)) for p in aggregate["parts"])
    aggregate["summary"] = {
        "n_parts": int(len(aggregate["parts"])),
        "n_records": int(n_records),
        "n_augmented": int(n_augmented),
        "n_failed": int(n_failed),
        "n_nodes": int(n_nodes),
        "n_mapped_nodes": int(n_mapped),
        "n_ambiguous_keys": int(n_amb),
        "mapping_coverage_pct": _fmt_pct(n_mapped, n_nodes),
        "phi_defined_pct_of_mapped": _fmt_pct(n_phi, n_mapped),
        "psi_defined_pct_of_mapped": _fmt_pct(n_psi, n_mapped),
    }

    report_out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"[TORSION] report={report_out}", flush=True)
    print(
        f"[TORSION] summary records={n_augmented}/{n_records} "
        f"mapping={aggregate['summary']['mapping_coverage_pct']:.2f}% "
        f"phi={aggregate['summary']['phi_defined_pct_of_mapped']:.2f}% "
        f"psi={aggregate['summary']['psi_defined_pct_of_mapped']:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()

