from __future__ import annotations

import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

from bridging.MD.prefetch_pdbs import ensure_pdb_cached
from bridging.utils.affinity import experimental_delta_g_kcalmol
from bridging.utils.dataset_rows import parse_chain_group, row_pdb_id
from bridging.utils.table import normalized_lookup


def _evenly_spaced_from_indices(indices: np.ndarray, n_pick: int) -> np.ndarray:
    if indices.size == 0:
        return np.array([], dtype=np.int64)
    if n_pick <= 0:
        raise ValueError("frames_per_complex must be > 0")
    if indices.size == 1:
        return np.repeat(indices, n_pick).astype(np.int64)
    pos = np.linspace(0, indices.size - 1, num=n_pick)
    pick = np.rint(pos).astype(np.int64)
    pick = np.clip(pick, 0, indices.size - 1)
    return indices[pick].astype(np.int64)


def _normalize_chain_string(raw: str | None) -> str | None:
    if raw is None:
        return None
    chains = parse_chain_group(raw)
    if not chains:
        return None
    return ",".join(chains)


def _explicit_chain_groups(row: dict) -> tuple[str | None, str | None]:
    lookup = normalized_lookup(row)
    if "ligandchains" in lookup and "receptorchains" in lookup:
        return str(row.get(lookup["ligandchains"])), str(row.get(lookup["receptorchains"]))
    if "chains1" in lookup and "chains2" in lookup:
        return str(row.get(lookup["chains1"])), str(row.get(lookup["chains2"]))
    return None, None


def _select_rows(dataset_csv: Path) -> tuple[list[dict], dict]:
    df = pd.read_csv(dataset_csv)
    selected: dict[str, dict] = {}
    stats = {
        "rows_total": int(len(df)),
        "rows_missing_pdb": 0,
        "rows_missing_chain_groups": 0,
        "rows_missing_dg": 0,
        "rows_duplicate_pdb_skipped": 0,
    }

    for row in df.to_dict("records"):
        pdb = row_pdb_id(row)
        if not pdb:
            stats["rows_missing_pdb"] += 1
            continue

        left_raw, right_raw = _explicit_chain_groups(row)
        left = _normalize_chain_string(left_raw)
        right = _normalize_chain_string(right_raw)
        if not left or not right:
            stats["rows_missing_chain_groups"] += 1
            continue

        dg = experimental_delta_g_kcalmol(row)
        if dg is None:
            stats["rows_missing_dg"] += 1
            continue

        pdb = str(pdb).upper()
        if pdb in selected:
            stats["rows_duplicate_pdb_skipped"] += 1
            continue

        selected[pdb] = {
            "pdb": pdb,
            "ligand": left,
            "receptor": right,
            "dG": float(dg),
        }

    rows = [selected[p] for p in sorted(selected)]
    stats["rows_selected_unique_pdb"] = int(len(rows))
    return rows, stats


def _build_ppb_row(
    *,
    pdb: str,
    source: str,
    ligand: str,
    receptor: str,
    dG: float,
    pdb_path: Path,
) -> dict:
    return {
        "pdb": pdb,
        "mutstr": np.nan,
        "source": source,
        "ligand": ligand,
        "receptor": receptor,
        "dG": float(dG),
        "pdb_path": str(pdb_path.resolve()),
    }


def _save_frame_pdb(traj: md.Trajectory, frame_index: int, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj[frame_index].save_pdb(str(out_path))


def build_ppb_inputs(
    *,
    dataset_csv: Path,
    md_root: Path,
    out_dir: Path,
    mode: str,
    frames_per_complex: int,
    drop_first_fraction: float,
    pdb_cache_root: Path,
    overwrite: bool,
) -> dict:
    rows, stats = _select_rows(dataset_csv)

    baseline_rows: list[dict] = []
    frame_rows: list[dict] = []
    skipped_missing_md: list[str] = []
    skipped_empty_traj: list[str] = []
    skipped_missing_raw_pdb: list[str] = []

    structures_root = out_dir / "structures"
    structures_root.mkdir(parents=True, exist_ok=True)

    for rec in rows:
        pdb = rec["pdb"]
        md_dir = md_root / pdb
        traj_path = md_dir / "traj_protein.nc"
        top_path = md_dir / "topology_protein.pdb"
        if not traj_path.exists() or not top_path.exists():
            skipped_missing_md.append(pdb)
            continue

        traj = md.load(str(traj_path), top=str(top_path))
        n_frames = int(traj.n_frames)
        if n_frames <= 0:
            skipped_empty_traj.append(pdb)
            continue
        frac = min(max(float(drop_first_fraction), 0.0), 0.95)
        start_idx = min(int(np.floor(frac * n_frames)), n_frames - 1)
        candidate_idx = np.arange(start_idx, n_frames, dtype=np.int64)

        if mode in {"baseline", "both"}:
            try:
                raw_pdb_path, _ = ensure_pdb_cached(pdb, cache_dir=Path(pdb_cache_root))
            except Exception:
                skipped_missing_raw_pdb.append(pdb)
            else:
                baseline_rows.append(
                    _build_ppb_row(
                        pdb=pdb,
                        source="RCSB",
                        ligand=rec["ligand"],
                        receptor=rec["receptor"],
                        dG=rec["dG"],
                        pdb_path=raw_pdb_path,
                    )
                )

        if mode in {"frame_aug", "both"}:
            frame_indices = _evenly_spaced_from_indices(candidate_idx, frames_per_complex)
            for j, frame_idx in enumerate(frame_indices.tolist()):
                source = f"F{j:03d}"
                frame_pdb_path = structures_root / "frames" / pdb / source / f"{pdb}.pdb"
                _save_frame_pdb(traj, int(frame_idx), frame_pdb_path, overwrite=overwrite)
                frame_rows.append(
                    _build_ppb_row(
                        pdb=pdb,
                        source=source,
                        ligand=rec["ligand"],
                        receptor=rec["receptor"],
                        dG=rec["dG"],
                        pdb_path=frame_pdb_path,
                    )
                )

    output = {
        "dataset_csv": str(dataset_csv.resolve()),
        "md_root": str(md_root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "mode": mode,
        "frames_per_complex": int(frames_per_complex),
        "drop_first_fraction": float(drop_first_fraction),
        "selected_stats": stats,
        "n_baseline_rows": int(len(baseline_rows)),
        "n_frame_rows": int(len(frame_rows)),
        "n_missing_md": int(len(skipped_missing_md)),
        "n_empty_traj": int(len(skipped_empty_traj)),
        "n_missing_raw_pdb": int(len(skipped_missing_raw_pdb)),
        "missing_md_pdb": skipped_missing_md[:50],
        "empty_traj_pdb": skipped_empty_traj[:50],
        "missing_raw_pdb": skipped_missing_raw_pdb[:50],
    }

    if baseline_rows:
        baseline_csv = out_dir / "ppb_baseline.csv"
        pd.DataFrame(baseline_rows).to_csv(baseline_csv, index=False)
        output["baseline_csv"] = str(baseline_csv.resolve())

    if frame_rows:
        frame_csv = out_dir / "ppb_frame_aug.csv"
        pd.DataFrame(frame_rows).to_csv(frame_csv, index=False)
        output["frame_aug_csv"] = str(frame_csv.resolve())

    report_path = out_dir / "ppb_prepare_report.json"
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["report_json"] = str(report_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare PPB-Affinity compatible CSV inputs from Bridging MD trajectories. "
            "Creates baseline (one frame per complex) and/or frame-augmented (many frames) datasets."
        )
    )
    parser.add_argument("--dataset", required=True, help="Bridging dataset CSV used for labels/chains")
    parser.add_argument("--md-root", required=True, help="MD output root directory with per-PDB folders")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated PDB frames + CSV files")
    parser.add_argument("--mode", choices=["baseline", "frame_aug", "both"], default="both")
    parser.add_argument("--frames-per-complex", type=int, default=120)
    parser.add_argument(
        "--drop-first-fraction",
        type=float,
        default=0.0,
        help="Fraction of early trajectory frames to discard before sampling (e.g. 0.17 to drop equil-like prefix).",
    )
    parser.add_argument(
        "--pdb-cache-root",
        default="~/scratch/pdb_cache",
        help="Directory for raw RCSB PDB files used by baseline PPB model.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = build_ppb_inputs(
        dataset_csv=Path(args.dataset),
        md_root=Path(args.md_root),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        frames_per_complex=args.frames_per_complex,
        drop_first_fraction=args.drop_first_fraction,
        pdb_cache_root=Path(args.pdb_cache_root).expanduser(),
        overwrite=bool(args.overwrite),
    )

    print(f"[PPB-PREP] baseline_rows={result.get('n_baseline_rows', 0)}")
    print(f"[PPB-PREP] frame_rows={result.get('n_frame_rows', 0)}")
    print(
        f"[PPB-PREP] missing_md={result.get('n_missing_md', 0)} "
        f"empty_traj={result.get('n_empty_traj', 0)} "
        f"missing_raw_pdb={result.get('n_missing_raw_pdb', 0)}"
    )
    if "baseline_csv" in result:
        print(f"[PPB-PREP] baseline_csv={result['baseline_csv']}")
    if "frame_aug_csv" in result:
        print(f"[PPB-PREP] frame_aug_csv={result['frame_aug_csv']}")
    print(f"[PPB-PREP] report_json={result['report_json']}")


if __name__ == "__main__":
    main()
