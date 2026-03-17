from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
from openmm import Context, CustomNonbondedForce, NonbondedForce, System, Vec3, VerletIntegrator, unit
from openmm.app import ForceField, Modeller, PDBFile

from bridging.MD.config import FORCEFIELD_FILES
from bridging.MD.simulate import build_system
from bridging.graphvae.common.chain_remap import _load_chain_sequences, build_raw_to_md_chain_map
from bridging.utils.dataset_rows import parse_chain_group

from ..common.config import FORCE_NODE_INPUT_FEATURES
from .md_dynamics import _build_residue_maps


ONE_4PI_EPS0 = 138.935456
COUL_GROUP = 11
LJ_GROUP = 12


def _bitmask(group: int) -> int:
    return 1 << int(group)


def _ordered_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys([str(x).strip().upper() for x in items if str(x).strip()]))


def remap_chain_groups_to_md(
    *,
    raw_pdb_path: Path,
    md_topology_pdb: Path,
    ligand_group: str,
    receptor_group: str,
) -> tuple[list[str], list[str], dict]:
    raw_lig = _ordered_unique(parse_chain_group(ligand_group))
    raw_rec = _ordered_unique(parse_chain_group(receptor_group))
    query_chains = _ordered_unique(raw_lig + raw_rec)
    chain_map, md_chain_order, report = build_raw_to_md_chain_map(
        raw_pdb_path,
        md_topology_pdb,
        query_chains=query_chains,
    )
    unresolved = [c for c in query_chains if c not in chain_map]
    if unresolved:
        raise RuntimeError(
            f"Could not remap query chains {unresolved}; available_md={md_chain_order} "
            f"query_map={report.get('query_mapped', {})}"
        )
    md_lig = _ordered_unique([chain_map[c] for c in raw_lig])
    md_rec = _ordered_unique([chain_map[c] for c in raw_rec])
    report = {
        **report,
        "raw_ligand_group": raw_lig,
        "raw_receptor_group": raw_rec,
        "md_ligand_group": md_lig,
        "md_receptor_group": md_rec,
        "md_chain_order": md_chain_order,
    }
    return md_lig, md_rec, report


def assess_force_query_compatibility(
    *,
    raw_pdb_path: Path,
    protein_topology_pdb: Path,
    ligand_group: str,
    receptor_group: str,
    full_topology_pdb: Path | None = None,
) -> dict:
    raw_lig = _ordered_unique(parse_chain_group(ligand_group))
    raw_rec = _ordered_unique(parse_chain_group(receptor_group))
    query_chains = _ordered_unique(raw_lig + raw_rec)

    raw_info = _load_chain_sequences(raw_pdb_path)
    protein_info = _load_chain_sequences(protein_topology_pdb)
    full_info = None
    if full_topology_pdb is not None and full_topology_pdb.exists():
        full_info = _load_chain_sequences(full_topology_pdb)

    def _missing(order: list[str], query: list[str]) -> list[str]:
        present = {str(x).strip().upper() for x in order}
        return [c for c in query if c not in present]

    raw_overlap = sorted(set(raw_lig).intersection(raw_rec))
    report = {
        "raw_ligand_group": raw_lig,
        "raw_receptor_group": raw_rec,
        "raw_query_chains": query_chains,
        "raw_chain_order": list(raw_info.order),
        "protein_chain_order": list(protein_info.order),
        "full_chain_order": [] if full_info is None else list(full_info.order),
        "missing_in_raw": _missing(list(raw_info.order), query_chains),
        "missing_in_protein": _missing(list(protein_info.order), query_chains),
        "missing_in_full": [] if full_info is None else _missing(list(full_info.order), query_chains),
        "raw_query_overlap": raw_overlap,
        "compatible": False,
        "compatibility_reason": "",
    }
    if raw_overlap:
        report["compatibility_reason"] = "raw_query_overlap"
        return report

    try:
        md_lig, md_rec, remap_report = remap_chain_groups_to_md(
            raw_pdb_path=raw_pdb_path,
            md_topology_pdb=protein_topology_pdb,
            ligand_group=ligand_group,
            receptor_group=receptor_group,
        )
    except Exception as exc:
        report["compatibility_reason"] = "query_remap_failed"
        report["error"] = repr(exc)
        return report

    overlap_after_remap = sorted(set(md_lig).intersection(md_rec))
    report.update(
        {
            "md_ligand_group": md_lig,
            "md_receptor_group": md_rec,
            "remap_report": remap_report,
            "overlap_after_remap": overlap_after_remap,
        }
    )
    if overlap_after_remap:
        report["compatibility_reason"] = "overlap_after_remap"
        return report

    report["compatible"] = True
    report["compatibility_reason"] = "compatible"
    return report


def _box_vectors_from_nm(box_nm: np.ndarray) -> tuple[Vec3, Vec3, Vec3]:
    box = np.asarray(box_nm, dtype=np.float64)
    if box.shape != (3, 3):
        raise ValueError(f"Expected box vectors shape (3, 3), got {box.shape}")
    return (
        Vec3(*box[0]) * unit.nanometer,
        Vec3(*box[1]) * unit.nanometer,
        Vec3(*box[2]) * unit.nanometer,
    )


def _load_analysis_system(topology_pdb: Path, box_vectors_nm: np.ndarray | None = None) -> tuple[PDBFile, System]:
    pdb = PDBFile(str(topology_pdb))
    if pdb.topology.getPeriodicBoxVectors() is None and box_vectors_nm is not None:
        pdb.topology.setPeriodicBoxVectors(_box_vectors_from_nm(box_vectors_nm))
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField(*FORCEFIELD_FILES)
    system = build_system(forcefield, modeller, allow_ignore_external_bonds=False)
    return pdb, system


def _extract_nonbonded_force(system: System) -> NonbondedForce:
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, NonbondedForce):
            return force
    raise RuntimeError("No NonbondedForce found in reconstructed system.")


def _custom_method_from_nonbonded(nonbonded: NonbondedForce) -> int:
    method = nonbonded.getNonbondedMethod()
    if method in (NonbondedForce.CutoffNonPeriodic, NonbondedForce.NoCutoff):
        return CustomNonbondedForce.NoCutoff if method == NonbondedForce.NoCutoff else CustomNonbondedForce.CutoffNonPeriodic
    return CustomNonbondedForce.CutoffPeriodic


def _configure_common_custom_force(custom: CustomNonbondedForce, nonbonded: NonbondedForce) -> None:
    custom.setNonbondedMethod(_custom_method_from_nonbonded(nonbonded))
    custom.setCutoffDistance(nonbonded.getCutoffDistance())
    if hasattr(nonbonded, "getUseSwitchingFunction") and nonbonded.getUseSwitchingFunction():
        custom.setUseSwitchingFunction(True)
        custom.setSwitchingDistance(nonbonded.getSwitchingDistance())
    custom.setUseLongRangeCorrection(False)
    for i in range(nonbonded.getNumExceptions()):
        a1, a2, *_ = nonbonded.getExceptionParameters(i)
        custom.addExclusion(int(a1), int(a2))


def build_interchain_force_system(
    *,
    source_system: System,
    source_topology,
    ligand_chain_ids: list[str],
    receptor_chain_ids: list[str],
) -> tuple[System, int, int]:
    source_nb = _extract_nonbonded_force(source_system)

    ligand_set = {str(c).strip().upper() for c in ligand_chain_ids}
    receptor_set = {str(c).strip().upper() for c in receptor_chain_ids}
    if not ligand_set or not receptor_set:
        raise RuntimeError("Empty ligand/receptor chain groups for force analysis.")

    chain_atoms: dict[str, list[int]] = {}
    for chain in source_topology.chains():
        cid = str(chain.id).strip().upper()
        atoms = [int(atom.index) for residue in chain.residues() for atom in residue.atoms()]
        chain_atoms[cid] = atoms

    lig_atoms = sorted({a for cid in ligand_set for a in chain_atoms.get(cid, [])})
    rec_atoms = sorted({a for cid in receptor_set for a in chain_atoms.get(cid, [])})
    if not lig_atoms or not rec_atoms:
        raise RuntimeError(
            f"Could not resolve chain atom sets for force analysis. lig={sorted(ligand_set)} rec={sorted(receptor_set)} "
            f"available={sorted(chain_atoms)}"
        )
    if set(lig_atoms).intersection(rec_atoms):
        raise RuntimeError(
            f"Ligand/receptor atom groups overlap after remapping. lig={sorted(ligand_set)} rec={sorted(receptor_set)}"
        )

    analysis = System()
    for i in range(source_system.getNumParticles()):
        analysis.addParticle(source_system.getParticleMass(i))

    coul = CustomNonbondedForce("ONE_4PI_EPS0*q1*q2/r")
    coul.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    coul.addPerParticleParameter("q")
    _configure_common_custom_force(coul, source_nb)
    coul.addInteractionGroup(lig_atoms, rec_atoms)
    coul.setForceGroup(COUL_GROUP)

    lj = CustomNonbondedForce(
        "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
        "sigma=0.5*(sigma1+sigma2);"
        "epsilon=sqrt(epsilon1*epsilon2)"
    )
    lj.addPerParticleParameter("sigma")
    lj.addPerParticleParameter("epsilon")
    _configure_common_custom_force(lj, source_nb)
    lj.addInteractionGroup(lig_atoms, rec_atoms)
    lj.setForceGroup(LJ_GROUP)

    for i in range(source_nb.getNumParticles()):
        charge, sigma, epsilon = source_nb.getParticleParameters(i)
        coul.addParticle([charge])
        lj.addParticle([sigma, epsilon])

    analysis.addForce(coul)
    analysis.addForce(lj)
    return analysis, COUL_GROUP, LJ_GROUP


def _context_for_force_analysis(
    *,
    topology_pdb: Path,
    box_vectors_nm: np.ndarray | None,
    ligand_chain_ids: list[str],
    receptor_chain_ids: list[str],
) -> tuple[Context, PDBFile]:
    pdb, source_system = _load_analysis_system(topology_pdb, box_vectors_nm=box_vectors_nm)
    analysis_system, coul_group, lj_group = build_interchain_force_system(
        source_system=source_system,
        source_topology=pdb.topology,
        ligand_chain_ids=ligand_chain_ids,
        receptor_chain_ids=receptor_chain_ids,
    )
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    context = Context(analysis_system, integrator)
    context._bridging_force_groups = (coul_group, lj_group)  # type: ignore[attr-defined]
    return context, pdb


def _set_context_frame(context: Context, traj: md.Trajectory, frame_idx: int) -> None:
    positions = np.asarray(traj.xyz[frame_idx], dtype=np.float64) * unit.nanometer
    context.setPositions(positions)
    if traj.unitcell_vectors is not None:
        box = np.asarray(traj.unitcell_vectors[frame_idx], dtype=np.float64)
        a = Vec3(*box[0]) * unit.nanometer
        b = Vec3(*box[1]) * unit.nanometer
        c = Vec3(*box[2]) * unit.nanometer
        context.setPeriodicBoxVectors(a, b, c)


def compute_node_interchain_force_features(
    *,
    traj: md.Trajectory,
    topology_pdb: Path,
    raw_pdb_path: Path,
    ligand_group: str,
    receptor_group: str,
    node_chain_id: list[str],
    node_position: list[int],
) -> tuple[np.ndarray, dict]:
    md_lig, md_rec, remap_report = remap_chain_groups_to_md(
        raw_pdb_path=raw_pdb_path,
        md_topology_pdb=topology_pdb,
        ligand_group=ligand_group,
        receptor_group=receptor_group,
    )

    context, pdb = _context_for_force_analysis(
        topology_pdb=topology_pdb,
        box_vectors_nm=None if traj.unitcell_vectors is None else traj.unitcell_vectors[0],
        ligand_chain_ids=md_lig,
        receptor_chain_ids=md_rec,
    )
    coul_group, lj_group = context._bridging_force_groups  # type: ignore[attr-defined]

    (
        residue_map,
        _ca_atom_map,
        _heavy_atom_map,
        _residue_chain_map,
        atom_residue_map,
        _protein_residue_keys,
        _all_heavy_atoms,
    ) = _build_residue_maps(traj.topology)

    node_keys = [(str(c).strip().upper(), int(p)) for c, p in zip(node_chain_id, node_position)]
    node_to_residue: list[int | None] = []
    residue_to_node: dict[int, int] = {}
    mapped_nodes = 0
    for idx, key in enumerate(node_keys):
        residue_idx = residue_map.get(key)
        node_to_residue.append(residue_idx)
        if residue_idx is not None:
            residue_to_node[int(residue_idx)] = idx
            mapped_nodes += 1

    n_nodes = len(node_keys)
    n_frames = int(traj.n_frames)
    coul_mag = np.zeros((n_frames, n_nodes), dtype=np.float64)
    lj_mag = np.zeros((n_frames, n_nodes), dtype=np.float64)

    force_unit = unit.kilojoule_per_mole / unit.nanometer
    for frame_idx in range(n_frames):
        _set_context_frame(context, traj, frame_idx)
        state_c = context.getState(getForces=True, groups=_bitmask(coul_group))
        state_lj = context.getState(getForces=True, groups=_bitmask(lj_group))
        atom_forces_c = np.asarray(state_c.getForces(asNumpy=True).value_in_unit(force_unit), dtype=np.float64)
        atom_forces_lj = np.asarray(state_lj.getForces(asNumpy=True).value_in_unit(force_unit), dtype=np.float64)

        residue_forces_c = np.zeros((len(residue_to_node), 3), dtype=np.float64)
        residue_forces_lj = np.zeros((len(residue_to_node), 3), dtype=np.float64)
        residue_slot_lookup = {res_idx: slot for slot, res_idx in enumerate(sorted(residue_to_node))}

        for atom_idx, residue_idx in atom_residue_map.items():
            node_idx = residue_to_node.get(int(residue_idx))
            if node_idx is None:
                continue
            slot = residue_slot_lookup[int(residue_idx)]
            residue_forces_c[slot] += atom_forces_c[int(atom_idx)]
            residue_forces_lj[slot] += atom_forces_lj[int(atom_idx)]

        sorted_residue_indices = sorted(residue_to_node)
        for slot, residue_idx in enumerate(sorted_residue_indices):
            node_idx = residue_to_node[int(residue_idx)]
            coul_mag[frame_idx, node_idx] = float(np.linalg.norm(residue_forces_c[slot]))
            lj_mag[frame_idx, node_idx] = float(np.linalg.norm(residue_forces_lj[slot]))

    out = np.stack(
        [
            np.mean(coul_mag, axis=0),
            np.std(coul_mag, axis=0),
            np.mean(lj_mag, axis=0),
            np.std(lj_mag, axis=0),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    stats = {
        "n_nodes": int(n_nodes),
        "n_mapped_nodes": int(mapped_nodes),
        "n_frames": int(n_frames),
        "ligand_group_md": md_lig,
        "receptor_group_md": md_rec,
        "remap_report": remap_report,
        "feature_names": list(FORCE_NODE_INPUT_FEATURES),
    }

    del context
    del pdb
    return out, stats
