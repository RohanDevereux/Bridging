import json
from pathlib import Path

from openmm import unit
from openmm.app import PDBFile, Topology

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _to_unit_positions(positions, pos_list):
    if hasattr(positions, "unit"):
        pos_unit = positions.unit
        return unit.Quantity([p.value_in_unit(pos_unit) for p in pos_list], pos_unit)
    return unit.Quantity(pos_list, unit.nanometer)

def write_atom_subset_topology_pdb(path, topology, positions, atom_indices):
    """Write a PDB for an arbitrary atom subset (keeps chain/residue/atom IDs)."""
    keep = set(int(i) for i in atom_indices)
    top_new = Topology()
    pos_new = []

    for chain in topology.chains():
        has_atoms = False
        for res in chain.residues():
            if any(a.index in keep for a in res.atoms()):
                has_atoms = True
                break
        if not has_atoms:
            continue
        new_chain = top_new.addChain(chain.id)
        for res in chain.residues():
            res_atoms = [a for a in res.atoms() if a.index in keep]
            if not res_atoms:
                continue
            new_res = top_new.addResidue(res.name, new_chain, res.id)
            for atom in res_atoms:
                top_new.addAtom(atom.name, atom.element, new_res, atom.id)
                pos_new.append(positions[atom.index])

    pos_new = _to_unit_positions(positions, pos_new)
    with open(path, "w") as f:
        PDBFile.writeFile(top_new, pos_new, f)

def write_ca_topology_pdb(path, topology, positions):
    """Write a PDB containing only CA atoms."""
    top_ca = Topology()
    pos_ca = []

    for chain in topology.chains():
        residues = []
        for res in chain.residues():
            if any(a.name == "CA" for a in res.atoms()):
                residues.append(res)
        if not residues:
            continue

        new_chain = top_ca.addChain(chain.id)
        for res in residues:
            new_res = top_ca.addResidue(res.name, new_chain, res.id)
            for atom in res.atoms():
                if atom.name == "CA":
                    top_ca.addAtom(atom.name, atom.element, new_res, atom.id)
                    pos_ca.append(positions[atom.index])

    pos_ca = _to_unit_positions(positions, pos_ca)

    with open(path, "w") as f:
        PDBFile.writeFile(top_ca, pos_ca, f)

def get_ca_atom_indices(topology):
    return [a.index for a in topology.atoms() if a.name == "CA"]

def get_protein_atom_indices(topology):
    """All atoms from residues that contain a CA atom (protein-only in this pipeline)."""
    out = []
    for residue in topology.residues():
        if not any(a.name == "CA" for a in residue.atoms()):
            continue
        out.extend([a.index for a in residue.atoms()])
    return out
