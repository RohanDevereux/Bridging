from pathlib import Path

from pdbfixer import PDBFixer
from openmm.app import Modeller, ForceField
from openmm.unit import nanometer, molar

from .config import FORCEFIELD_FILES, WATER_PADDING_NM, IONIC_STRENGTH_M

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def load_and_fix(pdb_source):
    pdb_path = Path(pdb_source)
    if pdb_path.exists():
        fixer = PDBFixer(filename=str(pdb_path))
    else:
        pdb_id = pdb_path.stem if pdb_path.suffix else str(pdb_source)
        fixer = PDBFixer(pdbid=str(pdb_id))

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    return fixer


def select_chains(topology, positions, chain_ids):
    modeller = Modeller(topology, positions)
    keep = set(chain_ids)
    delete = [c for c in modeller.topology.chains() if c.id not in keep]
    if delete:
        modeller.delete(delete)
    return modeller


def drop_non_protein_residues(modeller):
    delete = [r for r in modeller.topology.residues() if r.name not in STANDARD_AA]
    if delete:
        modeller.delete(delete)
    return modeller


def mark_disulfides(modeller):
    modeller.topology.createDisulfideBonds(modeller.positions)
    return modeller


def cysteine_residue_templates(topology):
    disulfide = set()
    for a1, a2 in topology.bonds():
        if (
            a1.name == "SG"
            and a2.name == "SG"
            and a1.residue.name == "CYS"
            and a2.residue.name == "CYS"
        ):
            disulfide.add(a1.residue)
            disulfide.add(a2.residue)

    templates = {}
    for residue in topology.residues():
        if residue.name == "CYS":
            templates[residue] = "CYX" if residue in disulfide else "CYS"
    return templates


def cysteine_variants(topology):
    disulfide = set()
    for a1, a2 in topology.bonds():
        if (
            a1.name == "SG"
            and a2.name == "SG"
            and a1.residue.name == "CYS"
            and a2.residue.name == "CYS"
        ):
            disulfide.add(a1.residue)
            disulfide.add(a2.residue)

    variants = []
    for residue in topology.residues():
        if residue.name == "CYS":
            variants.append("CYX" if residue in disulfide else "CYS")
        else:
            variants.append(None)
    return variants


def solvate(modeller, ph, variants=None):
    forcefield = ForceField(*FORCEFIELD_FILES)
    modeller.addHydrogens(forcefield, pH=ph, variants=variants)
    modeller.addSolvent(
        forcefield,
        padding=WATER_PADDING_NM * nanometer,
        ionicStrength=IONIC_STRENGTH_M * molar,
    )
    return forcefield, modeller
