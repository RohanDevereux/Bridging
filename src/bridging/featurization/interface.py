import numpy as np


def _ca_atoms_for_chains(traj, chain_ids):
    chain_ids = set(chain_ids)
    ca = []
    for atom in traj.topology.atoms:
        if atom.name == "CA" and atom.residue.chain.id in chain_ids:
            ca.append(atom.index)
    return np.array(ca, dtype=int)


def select_interface_atoms(traj, chains_1, chains_2, n):
    c1 = list(chains_1)
    c2 = list(chains_2)

    a1 = _ca_atoms_for_chains(traj, c1)
    a2 = _ca_atoms_for_chains(traj, c2)

    xyz0 = traj.xyz[0]
    X = xyz0[a1]
    Y = xyz0[a2]

    d12 = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    min1 = d12.min(axis=1)
    min2 = d12.min(axis=0)

    idx1 = a1[np.argsort(min1)[:n]]
    idx2 = a2[np.argsort(min2)[:n]]

    return idx1, idx2
