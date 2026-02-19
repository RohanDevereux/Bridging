from pathlib import Path

import mdtraj as md


def load_ca_trajectory(out_dir):
    out_dir = Path(out_dir)

    legacy_h5 = out_dir / "traj_ca.h5"
    if legacy_h5.exists():
        return md.load_hdf5(str(legacy_h5))

    protein_nc = out_dir / "traj_protein.nc"
    protein_top = out_dir / "topology_protein.pdb"
    if protein_nc.exists() and protein_top.exists():
        traj = md.load(str(protein_nc), top=str(protein_top))
        ca_idx = traj.topology.select("name CA")
        if ca_idx.size == 0:
            raise ValueError(f"No CA atoms in {protein_top}")
        return traj.atom_slice(ca_idx)

    full_nc = out_dir / "traj_full.nc"
    full_top = out_dir / "topology_full.pdb"
    if full_nc.exists() and full_top.exists():
        traj = md.load(str(full_nc), top=str(full_top))
        ca_idx = traj.topology.select("name CA")
        if ca_idx.size == 0:
            raise ValueError(f"No CA atoms in {full_top}")
        return traj.atom_slice(ca_idx)

    raise FileNotFoundError(
        f"No supported trajectory files found in {out_dir} "
        "(expected traj_ca.h5 or traj_protein.nc/topology_protein.pdb or traj_full.nc/topology_full.pdb)."
    )
