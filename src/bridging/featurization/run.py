import json

import numpy as np

from .config import N_INTERFACE, STRIDE, D0_NM, K_NM, DTYPE
from .paths import features_dir
from .load_md import load_ca_trajectory
from .interface import select_interface_atoms
from .contact_map import soft_contact_maps
from ..MD.paths import MD_OUT_DIR


def run_all():
    for out_dir in sorted(MD_OUT_DIR.glob("*")):
        if not out_dir.is_dir():
            continue
        pdb_id = out_dir.name

        meta_path = out_dir / "meta.json"
        traj_path = out_dir / "traj_ca.h5"
        top_path = out_dir / "topology_ca.pdb"
        if not (meta_path.exists() and traj_path.exists() and top_path.exists()):
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chains_1 = meta["chains_1"]
        chains_2 = meta["chains_2"]

        fdir = features_dir(pdb_id)
        fdir.mkdir(parents=True, exist_ok=True)

        out_file = fdir / "contact_maps_fp16.npy"
        if out_file.exists():
            continue

        traj = load_ca_trajectory(out_dir)
        idx1, idx2 = select_interface_atoms(traj, chains_1, chains_2, N_INTERFACE)
        C = soft_contact_maps(traj, idx1, idx2, STRIDE, D0_NM, K_NM, dtype=DTYPE)

        np.save(out_file, C)

        feat_meta = {
            "pdb_id": pdb_id,
            "chains_1": chains_1,
            "chains_2": chains_2,
            "N_INTERFACE": N_INTERFACE,
            "STRIDE": STRIDE,
            "D0_NM": D0_NM,
            "K_NM": K_NM,
            "frames": int(C.shape[0]),
        }
        (fdir / "meta.json").write_text(json.dumps(feat_meta, indent=2), encoding="utf-8")
        print(f"[OK] featurized {pdb_id} -> {out_file.name} shape={C.shape}")


if __name__ == "__main__":
    run_all()
