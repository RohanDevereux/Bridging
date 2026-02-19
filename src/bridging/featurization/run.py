import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    N_INTERFACE,
    STRIDE,
    D0_NM,
    K_NM,
    DTYPE,
    INTERFACE_K_FRAMES,
    DIST_CLIP_NM,
    USE_LOG_DIST,
    FEATURES_BASENAME,
)
from .load_md import load_ca_trajectory
from .interface import select_interface_atoms
from .contact_map import contact_distance_channels
from ..MD.paths import MD_OUT_DIR
from ..utils.dataset_rows import parse_chain_group, row_chain_groups, row_pdb_id


def _dataset_chain_lookup(dataset_path):
    if dataset_path is None:
        return {}
    df = pd.read_csv(dataset_path)
    lookup = {}
    for row in df.to_dict("records"):
        pdb_id = row_pdb_id(row)
        if not pdb_id:
            continue

        chains_1, chains_2 = row_chain_groups(row)
        c1 = parse_chain_group(chains_1)
        c2 = parse_chain_group(chains_2)
        if c1 and c2 and pdb_id not in lookup:
            lookup[pdb_id] = (c1, c2)
    return lookup


def _resolve_chain_groups(meta, dataset_lookup, pdb_id):
    c1 = parse_chain_group(meta.get("chains_1"))
    c2 = parse_chain_group(meta.get("chains_2"))
    if c1 and c2:
        return c1, c2

    c = dataset_lookup.get(pdb_id.upper())
    if c is not None:
        return c

    return None, None


def _candidate_md_dirs(md_root):
    md_root = Path(md_root)
    names = ("traj_ca.h5", "traj_protein.nc", "traj_full.nc")
    dirs = set()
    for name in names:
        for p in md_root.rglob(name):
            if (p.parent / "DONE").exists():
                dirs.add(p.parent)
    return sorted(dirs)


def run_all(md_root=None, dataset_path=None, overwrite=False):
    root = Path(md_root) if md_root else MD_OUT_DIR
    dataset_lookup = _dataset_chain_lookup(dataset_path)
    out_dirs = _candidate_md_dirs(root)

    total = len(out_dirs)
    if total == 0:
        print(f"[FEAT] no trajectory outputs found under {root}")
        return

    print(f"[FEAT] md_root={root} complexes_with_traj={total}")
    if dataset_path:
        print(f"[FEAT] using dataset chain fallback: {dataset_path}")

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for out_dir in out_dirs:
        pdb_id = out_dir.name
        meta_path = out_dir / "meta.json"
        has_legacy = (out_dir / "traj_ca.h5").exists()
        has_protein = (out_dir / "traj_protein.nc").exists() and (out_dir / "topology_protein.pdb").exists()
        has_full = (out_dir / "traj_full.nc").exists() and (out_dir / "topology_full.pdb").exists()
        if not meta_path.exists() or not (has_legacy or has_protein or has_full):
            skip_count += 1
            continue

        fdir = out_dir / "features"
        fdir.mkdir(parents=True, exist_ok=True)
        out_file = fdir / FEATURES_BASENAME
        if out_file.exists() and not overwrite:
            skip_count += 1
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            chains_1, chains_2 = _resolve_chain_groups(meta, dataset_lookup, pdb_id)
            if not chains_1 or not chains_2:
                raise ValueError("missing chain groups (chains_1/chains_2 not found in meta or dataset)")

            traj = load_ca_trajectory(out_dir)
            idx1, idx2 = select_interface_atoms(
                traj,
                chains_1,
                chains_2,
                N_INTERFACE,
                method="stable",
                k_frames=INTERFACE_K_FRAMES,
                d0_nm=D0_NM,
                k_nm=K_NM,
            )
            C = contact_distance_channels(
                traj,
                idx1,
                idx2,
                STRIDE,
                D0_NM,
                K_NM,
                d_clip_nm=DIST_CLIP_NM,
                use_log_dist=USE_LOG_DIST,
                dtype=DTYPE,
            )

            np.save(out_file, C)
            feat_meta = {
                "pdb_id": pdb_id,
                "chains_1": "".join(chains_1),
                "chains_2": "".join(chains_2),
                "N_INTERFACE": N_INTERFACE,
                "STRIDE": STRIDE,
                "D0_NM": D0_NM,
                "K_NM": K_NM,
                "DIST_CLIP_NM": DIST_CLIP_NM,
                "USE_LOG_DIST": USE_LOG_DIST,
                "INTERFACE_K_FRAMES": INTERFACE_K_FRAMES,
                "INTERFACE_METHOD": "stable",
                "channels": ["soft_contact", "dist_log" if USE_LOG_DIST else "dist_lin"],
                "frames": int(C.shape[0]),
            }
            (fdir / "meta.json").write_text(json.dumps(feat_meta, indent=2), encoding="utf-8")
            ok_count += 1
            print(f"[OK] featurized {pdb_id} -> {out_file.name} shape={C.shape}")
        except Exception as exc:
            fail_count += 1
            (fdir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[FAIL] featurize {pdb_id}: {exc}")

    print(
        f"[FEAT] done complexes_with_traj={total} "
        f"ok={ok_count} skipped={skip_count} fail={fail_count}"
    )


def main():
    parser = argparse.ArgumentParser(description="Featurize completed MD outputs into contact/distance arrays.")
    parser.add_argument("--md-root", help="Root directory of MD outputs (defaults to generatedData/MD)")
    parser.add_argument(
        "--dataset",
        help="Optional dataset CSV used to recover chain groups when MD meta lacks chains_1/chains_2",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute features even if they exist")
    args = parser.parse_args()
    run_all(md_root=args.md_root, dataset_path=args.dataset, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
