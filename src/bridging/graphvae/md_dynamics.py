from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np


def _sample_frame_indices(n_frames: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or max_frames <= 0 or n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)
    pos = np.linspace(0, n_frames - 1, max_frames)
    return np.rint(pos).astype(np.int64)


def load_full_md_trajectory(md_dir: Path, max_frames: int | None = None) -> md.Trajectory:
    traj_path = md_dir / "traj_full.nc"
    top_path = md_dir / "topology_full.pdb"
    if not traj_path.exists() or not top_path.exists():
        raise FileNotFoundError(f"Missing full trajectory files in {md_dir}")
    traj = md.load(str(traj_path), top=str(top_path))
    idx = _sample_frame_indices(int(traj.n_frames), max_frames)
    return traj[idx]


def _chain_id(chain) -> str:
    for attr in ("chain_id", "id"):
        if hasattr(chain, attr):
            value = getattr(chain, attr)
            if value is not None and str(value).strip() != "":
                return str(value).strip().upper()
    return str(getattr(chain, "index")).strip().upper()


def _build_residue_maps(topology) -> tuple[dict, dict, dict, dict, dict, set[tuple[str, int]], list[int]]:
    residue_map: dict[tuple[str, int], int] = {}
    ca_atom_map: dict[tuple[str, int], int] = {}
    heavy_atom_map: dict[tuple[str, int], list[int]] = {}
    residue_chain_map: dict[int, str] = {}
    atom_residue_map: dict[int, int] = {}
    protein_residue_keys: set[tuple[str, int]] = set()
    all_heavy_atoms: list[int] = []

    for residue in topology.residues:
        atoms = list(residue.atoms)
        chain_key = _chain_id(residue.chain)
        residue_chain_map[int(residue.index)] = chain_key
        for atom in atoms:
            atom_residue_map[int(atom.index)] = int(residue.index)
        ca_atoms = [a for a in atoms if a.name == "CA"]
        if residue.is_protein and ca_atoms:
            protein_residue_keys.add((chain_key, int(residue.resSeq)))
        if not ca_atoms:
            continue
        key = (chain_key, int(residue.resSeq))
        residue_map[key] = int(residue.index)
        ca_atom_map[key] = int(ca_atoms[0].index)
        heavy = [int(a.index) for a in atoms if a.element is not None and a.element.symbol != "H"]
        heavy_atom_map[key] = heavy
        all_heavy_atoms.extend(heavy)
    return (
        residue_map,
        ca_atom_map,
        heavy_atom_map,
        residue_chain_map,
        atom_residue_map,
        protein_residue_keys,
        sorted(set(all_heavy_atoms)),
    )


def _water_oxygen_indices(topology) -> tuple[list[int], dict[int, int]]:
    oxy = []
    atom_to_water_res: dict[int, int] = {}
    for atom in topology.atoms:
        if not atom.residue.is_water:
            continue
        if atom.element is None or atom.element.symbol != "O":
            continue
        idx = int(atom.index)
        oxy.append(idx)
        atom_to_water_res[idx] = int(atom.residue.index)
    return oxy, atom_to_water_res


def _compute_rmsf_ca(traj: md.Trajectory, node_keys: list[tuple[str, int]], ca_atom_map: dict) -> np.ndarray:
    out = np.full((len(node_keys),), np.nan, dtype=np.float32)
    if traj.n_frames < 2:
        return out
    ca_global = sorted(set(ca_atom_map.values()))
    if len(ca_global) >= 2:
        aligned = traj.superpose(traj, frame=0, atom_indices=ca_global)
    else:
        aligned = traj

    for i, key in enumerate(node_keys):
        ca_idx = ca_atom_map.get(key)
        if ca_idx is None:
            continue
        xyz = aligned.xyz[:, ca_idx, :]
        mean_xyz = np.mean(xyz, axis=0, keepdims=True)
        rmsf = np.sqrt(np.mean(np.sum((xyz - mean_xyz) ** 2, axis=1)))
        out[i] = float(rmsf)
    return out


def _compute_water_counts(
    traj: md.Trajectory,
    node_keys: list[tuple[str, int]],
    heavy_atom_map: dict,
    water_oxygen: list[int],
    atom_to_water_res: dict[int, int],
    cutoff_nm: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    mean_out = np.full((len(node_keys),), np.nan, dtype=np.float32)
    std_out = np.full((len(node_keys),), np.nan, dtype=np.float32)
    if not water_oxygen:
        mean_out[:] = 0.0
        std_out[:] = 0.0
        return mean_out, std_out

    for i, key in enumerate(node_keys):
        query_atoms = heavy_atom_map.get(key, [])
        if not query_atoms:
            continue
        neigh = md.compute_neighbors(
            traj,
            cutoff=cutoff_nm,
            query_indices=query_atoms,
            haystack_indices=water_oxygen,
        )
        counts = np.zeros((traj.n_frames,), dtype=np.float32)
        for f, atom_idx_arr in enumerate(neigh):
            if len(atom_idx_arr) == 0:
                continue
            water_res = {atom_to_water_res[int(a)] for a in atom_idx_arr.tolist() if int(a) in atom_to_water_res}
            counts[f] = float(len(water_res))
        mean_out[i] = float(np.mean(counts))
        std_out[i] = float(np.std(counts))
    return mean_out, std_out


def _compute_edge_contact_stats(
    traj: md.Trajectory,
    edge_index: np.ndarray,
    node_keys: list[tuple[str, int]],
    residue_map: dict,
    include_distance_stats: bool,
    cutoff_nm: float = 0.8,
) -> np.ndarray:
    n_edges = int(edge_index.shape[1])
    out_dim = 3 if include_distance_stats else 1
    out = np.full((n_edges, out_dim), np.nan, dtype=np.float32)

    contacts: list[tuple[int, int]] = []
    edge_slots: list[int] = []
    for e in range(n_edges):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        key_i = node_keys[i]
        key_j = node_keys[j]
        res_i = residue_map.get(key_i)
        res_j = residue_map.get(key_j)
        if res_i is None or res_j is None:
            continue
        contacts.append((int(res_i), int(res_j)))
        edge_slots.append(e)

    if not contacts:
        return out

    dist, _ = md.compute_contacts(traj, contacts=contacts, scheme="closest-heavy")
    for slot, e in enumerate(edge_slots):
        d = dist[:, slot]
        out[e, 0] = float(np.mean(d <= cutoff_nm))
        if include_distance_stats:
            out[e, 1] = float(np.mean(d))
            out[e, 2] = float(np.std(d))
    return out


def _compute_chain_neighbor_counts(
    traj: md.Trajectory,
    node_keys: list[tuple[str, int]],
    residue_map: dict,
    heavy_atom_map: dict,
    atom_residue_map: dict[int, int],
    residue_chain_map: dict[int, str],
    all_heavy_atoms: list[int],
    cutoff_nm: float = 0.8,
) -> np.ndarray:
    out = np.full((len(node_keys), 2), np.nan, dtype=np.float32)
    if not all_heavy_atoms or traj.n_frames < 1:
        return out

    frame0 = traj[0]
    for i, key in enumerate(node_keys):
        query_atoms = heavy_atom_map.get(key, [])
        own_residue = residue_map.get(key)
        if own_residue is None or not query_atoms:
            continue

        neigh = md.compute_neighbors(
            frame0,
            cutoff=cutoff_nm,
            query_indices=query_atoms,
            haystack_indices=all_heavy_atoms,
        )[0]
        touched: set[int] = set()
        n_same = 0
        n_other = 0
        own_chain = str(key[0]).strip().upper()
        for atom_idx in neigh.tolist():
            residue_idx = atom_residue_map.get(int(atom_idx))
            if residue_idx is None or residue_idx == own_residue or residue_idx in touched:
                continue
            touched.add(residue_idx)
            chain_id = residue_chain_map.get(residue_idx, "")
            if chain_id == own_chain:
                n_same += 1
            else:
                n_other += 1
        out[i, 0] = float(n_same)
        out[i, 1] = float(n_other)
    return out


def compute_dynamic_features(
    *,
    traj: md.Trajectory,
    node_chain_id: list[str],
    node_position: list[int],
    edge_index: np.ndarray,
    include_distance_stats: bool = False,
) -> dict[str, np.ndarray]:
    topology = traj.topology
    (
        residue_map,
        ca_atom_map,
        heavy_atom_map,
        residue_chain_map,
        atom_residue_map,
        protein_residue_keys,
        all_heavy_atoms,
    ) = _build_residue_maps(topology)
    node_keys = [(str(c).strip().upper(), int(p)) for c, p in zip(node_chain_id, node_position)]

    water_oxygen, atom_to_water_res = _water_oxygen_indices(topology)
    water_mean, water_std = _compute_water_counts(
        traj,
        node_keys,
        heavy_atom_map,
        water_oxygen,
        atom_to_water_res,
    )
    rmsf = _compute_rmsf_ca(traj, node_keys, ca_atom_map)
    structural_context = _compute_chain_neighbor_counts(
        traj=traj,
        node_keys=node_keys,
        residue_map=residue_map,
        heavy_atom_map=heavy_atom_map,
        atom_residue_map=atom_residue_map,
        residue_chain_map=residue_chain_map,
        all_heavy_atoms=all_heavy_atoms,
    )

    node_dynamic = np.stack(
        [
            np.log1p(np.clip(water_mean, a_min=0.0, a_max=None)),
            water_std,
            rmsf,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    edge_dynamic = _compute_edge_contact_stats(
        traj,
        edge_index=edge_index,
        node_keys=node_keys,
        residue_map=residue_map,
        include_distance_stats=bool(include_distance_stats),
    ).astype(np.float32, copy=False)
    return {
        "node_dynamic": node_dynamic,
        "edge_dynamic": edge_dynamic,
        "node_structural_context": structural_context,
        "protein_residue_keys": protein_residue_keys,
    }
