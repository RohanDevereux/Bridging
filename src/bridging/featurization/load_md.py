import mdtraj as md


def load_ca_trajectory(out_dir):
    h5 = str(out_dir / "traj_ca.h5")
    return md.load_hdf5(h5)
