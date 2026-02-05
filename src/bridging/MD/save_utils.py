import json
from pathlib import Path
from openmm.app import PDBFile, Topology

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")

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

    with open(path, "w") as f:
        PDBFile.writeFile(top_ca, pos_ca, f)

def get_ca_atom_indices(topology):
    return [a.index for a in topology.atoms() if a.name == "CA"]
