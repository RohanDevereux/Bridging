import numpy as np


def _chain_id(chain):
    if hasattr(chain, "chain_id") and chain.chain_id:
        return chain.chain_id
    if hasattr(chain, "id") and chain.id:
        return chain.id
    return str(getattr(chain, "index", ""))


def _sort_by_residue_index(traj, atom_indices):
    return np.array(
        sorted(atom_indices, key=lambda idx: traj.topology.atom(int(idx)).residue.index),
        dtype=int,
    )


def _assert_has_atoms(a1, a2, c1, c2, traj):
    if a1.size == 0 or a2.size == 0:
        chain_ids = [_chain_id(c) for c in traj.topology.chains]
        unique = []
        for cid in chain_ids:
            if cid not in unique:
                unique.append(cid)
        raise ValueError(
            f"No CA atoms found for chains_1={c1} or chains_2={c2}. "
            f"Available chain IDs: {unique}"
        )


def _ca_atoms_for_chains(traj, chain_ids):
    chain_ids = set(chain_ids)
    ca = []
    for atom in traj.topology.atoms:
        if atom.name != "CA":
            continue
        chain_id = _chain_id(atom.residue.chain)
        if chain_id in chain_ids:
            ca.append(atom.index)
    return np.array(ca, dtype=int)


def select_interface_atoms(
    traj,
    chains_1,
    chains_2,
    n,
    *,
    method="stable",
    k_frames=50,
    d0_nm=0.8,
    k_nm=10.0,
):
    c1 = list(chains_1)
    c2 = list(chains_2)

    a1 = _ca_atoms_for_chains(traj, c1)
    a2 = _ca_atoms_for_chains(traj, c2)

    _assert_has_atoms(a1, a2, c1, c2, traj)
    if a1.size < n or a2.size < n:
        raise ValueError(
            f"Not enough CA atoms for requested N_INTERFACE={n} "
            f"(chains_1={a1.size}, chains_2={a2.size})."
        )

    if method in ("frame0", "closest"):
        xyz0 = traj.xyz[0]
        X = xyz0[a1]
        Y = xyz0[a2]

        d12 = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
        min1 = d12.min(axis=1)
        min2 = d12.min(axis=0)

        idx1 = a1[np.argsort(min1)[:n]]
        idx2 = a2[np.argsort(min2)[:n]]
        return _sort_by_residue_index(traj, idx1), _sort_by_residue_index(traj, idx2)

    if method != "stable":
        raise ValueError(f"Unknown interface selection method: {method}")

    t = min(int(k_frames), traj.n_frames)
    xyz = traj.xyz[:t]
    X = xyz[:, a1, :]
    Y = xyz[:, a2, :]

    diff = X[:, :, None, :] - Y[:, None, :, :]
    d12 = np.sqrt((diff * diff).sum(-1))
    c = 1.0 / (1.0 + np.exp(-k_nm * (d0_nm - d12)))

    score1 = c.max(axis=2).mean(axis=0)
    score2 = c.max(axis=1).mean(axis=0)

    idx1 = a1[np.argsort(-score1)[:n]]
    idx2 = a2[np.argsort(-score2)[:n]]

    return _sort_by_residue_index(traj, idx1), _sort_by_residue_index(traj, idx2)
