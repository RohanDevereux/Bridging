import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

from bridging.MD.paths import DATA_CSV, PPB_DATA_CSV, PDB_CACHE_DIR, resolve_dataset
from bridging.featurization.config import (
    N_INTERFACE,
    D0_NM,
    K_NM,
    DTYPE,
    INTERFACE_K_FRAMES,
    DIST_CLIP_NM,
    USE_LOG_DIST,
)
from bridging.featurization.contact_map import contact_distance_channels
from bridging.featurization.interface import select_interface_atoms
from bridging.featurization.load_md import load_ca_trajectory


def _parse_chain_group(value):
    if pd.isna(value):
        return ""
    cleaned = "".join([c for c in str(value) if c.isalnum()])
    return cleaned


def _chain_groups(row):
    normalized = {str(k).strip().lower(): k for k in row.keys()}
    if "chains_1" in normalized and "chains_2" in normalized:
        c1 = _parse_chain_group(row[normalized["chains_1"]])
        c2 = _parse_chain_group(row[normalized["chains_2"]])
        return c1, c2
    if "ligand chains" in normalized and "receptor chains" in normalized:
        c1 = _parse_chain_group(row[normalized["ligand chains"]])
        c2 = _parse_chain_group(row[normalized["receptor chains"]])
        return c1, c2
    if "ligand_chains" in normalized and "receptor_chains" in normalized:
        c1 = _parse_chain_group(row[normalized["ligand_chains"]])
        c2 = _parse_chain_group(row[normalized["receptor_chains"]])
        return c1, c2
    if "complex_pdb" in normalized:
        left = str(row[normalized["complex_pdb"]]).split("_", 1)[-1]
        left, right = left.split(":")
        return left, right
    raise ValueError("No chain columns found (expected Chains_1/Chains_2, Ligand/Receptor Chains, or complex_pdb).")


def _resolve_dataset_row(dataset_path, index=None, pdb_id=None):
    dataset_path = resolve_dataset(dataset_path, DATA_CSV)
    df = pd.read_csv(dataset_path)
    if index is not None:
        row = df.iloc[index]
        return row, dataset_path
    if pdb_id:
        if "PDB" in df.columns:
            series = df["PDB"]
        elif "PDB_ID" in df.columns:
            series = df["PDB_ID"]
        else:
            raise ValueError("Dataset missing PDB column (expected PDB or PDB_ID).")
        matches = df[series.astype(str).str.upper() == pdb_id.upper()]
        if matches.empty:
            raise ValueError(f"PDB {pdb_id} not found in dataset.")
        return matches.iloc[0], dataset_path
    raise ValueError("Provide --index or --pdb-id when using --dataset.")


def _load_traj_from_pdb(pdb_path):
    traj = md.load(str(pdb_path))
    ca_idx = [a.index for a in traj.topology.atoms if a.name == "CA"]
    if not ca_idx:
        raise ValueError("No CA atoms found in PDB.")
    traj = traj.atom_slice(ca_idx)
    return traj[:1]


def _load_traj_from_md_out(out_dir):
    out_dir = Path(out_dir)
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Missing meta.json in {out_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chains_1 = meta.get("chains_1")
    chains_2 = meta.get("chains_2")
    if not chains_1 or not chains_2:
        raise ValueError("meta.json missing chains_1/chains_2")
    traj = load_ca_trajectory(out_dir)
    return traj[:1], chains_1, chains_2


def featurize_first_frame(
    traj,
    chains_1,
    chains_2,
    out_path=None,
    print_values=True,
    *,
    method="stable",
    k_frames=INTERFACE_K_FRAMES,
    dist_clip_nm=DIST_CLIP_NM,
    use_log_dist=USE_LOG_DIST,
):
    idx1, idx2 = select_interface_atoms(
        traj,
        chains_1,
        chains_2,
        N_INTERFACE,
        method=method,
        k_frames=k_frames,
        d0_nm=D0_NM,
        k_nm=K_NM,
    )
    X = contact_distance_channels(
        traj,
        idx1,
        idx2,
        stride=1,
        d0_nm=D0_NM,
        k_nm=K_NM,
        d_clip_nm=dist_clip_nm,
        use_log_dist=use_log_dist,
        dtype=DTYPE,
    )
    C0 = X[0]
    print(f"[FEAT] feature shape={C0.shape} dtype={C0.dtype}")
    for ch in range(C0.shape[0]):
        ch_min = C0[ch].min()
        ch_max = C0[ch].max()
        ch_mean = C0[ch].mean()
        print(f"[FEAT] ch{ch} min={ch_min:.4f} max={ch_max:.4f} mean={ch_mean:.4f}")
    if print_values:
        print("[FEAT] preview (first 5x5 per channel):")
        for ch in range(C0.shape[0]):
            print(f"[FEAT] ch{ch} preview:")
            print(C0[ch][:5, :5])
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, C0)
        print(f"[FEAT] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Featurize first frame for a single complex.")
    parser.add_argument("--dataset", help="CSV path to use")
    parser.add_argument("--index", type=int, help="0-based row index in dataset")
    parser.add_argument("--pdb-id", help="PDB id to select from dataset")
    parser.add_argument("--pdb", help="PDB id or file path (bypass dataset)")
    parser.add_argument("--chains1", help="Chain IDs for partner 1 (e.g. AB)")
    parser.add_argument("--chains2", help="Chain IDs for partner 2 (e.g. CD)")
    parser.add_argument("--md-out", help="Use existing MD output dir (traj_ca.h5 + meta.json)")
    parser.add_argument("--out", help="Optional .npy output path for the feature map")
    parser.add_argument(
        "--method",
        default="stable",
        choices=["stable", "frame0", "closest"],
        help="Interface selection method",
    )
    parser.add_argument("--k-frames", type=int, default=INTERFACE_K_FRAMES, help="Frames for stable selection")
    parser.add_argument("--dist-clip-nm", type=float, default=DIST_CLIP_NM, help="Distance clip (nm)")
    parser.add_argument("--no-log-dist", action="store_true", help="Use linear distance channel")
    parser.add_argument("--no-preview", action="store_true", help="Disable printing preview values")
    parser.add_argument(
        "--ppb",
        action="store_true",
        help="Use PPB_Affinity_TCR_pMHC.csv as default dataset",
    )
    args = parser.parse_args()

    if args.md_out:
        traj, chains_1, chains_2 = _load_traj_from_md_out(args.md_out)
    else:
        chains_1 = args.chains1
        chains_2 = args.chains2
        pdb_path = None

        if args.pdb:
            path = Path(args.pdb)
            if path.exists():
                pdb_path = path
                pdb_id = path.stem
            else:
                pdb_id = args.pdb.upper()
                pdb_path = PDB_CACHE_DIR / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        else:
            default_path = PPB_DATA_CSV if args.ppb else DATA_CSV
            row, dataset_path = _resolve_dataset_row(args.dataset or default_path, args.index, args.pdb_id)
            pdb_id = str(row.get("PDB") or row.get("PDB_ID") or "").upper()
            if not pdb_id:
                raise ValueError("Dataset row missing PDB.")
            if not chains_1 or not chains_2:
                chains_1, chains_2 = _chain_groups(row)
            pdb_path = PDB_CACHE_DIR / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                raise FileNotFoundError(f"PDB file not found in cache: {pdb_path}")
            print(f"[INFO] dataset={dataset_path} pdb={pdb_id} chains1={chains_1} chains2={chains_2}")

        if not chains_1 or not chains_2:
            raise ValueError("Provide --chains1 and --chains2, or use a dataset with chain columns.")

        traj = _load_traj_from_pdb(pdb_path)

    featurize_first_frame(
        traj,
        chains_1,
        chains_2,
        out_path=args.out,
        print_values=not args.no_preview,
        method=args.method,
        k_frames=args.k_frames,
        dist_clip_nm=args.dist_clip_nm,
        use_log_dist=not args.no_log_dist,
    )


if __name__ == "__main__":
    main()
