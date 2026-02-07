from pathlib import Path

from pdbfixer import PDBFixer
from openmm import unit
from openmm.app import Modeller, ForceField
from openmm.app.element import hydrogen
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


def strip_hydrogens(modeller):
    h_atoms = [
        atom for atom in modeller.topology.atoms()
        if atom.element is not None and atom.element == hydrogen
    ]
    if h_atoms:
        modeller.delete(h_atoms)
    return modeller


def _distance_nm(positions, atom1, atom2):
    p1 = positions[atom1.index].value_in_unit(unit.nanometer)
    p2 = positions[atom2.index].value_in_unit(unit.nanometer)
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _remove_bonds(topology, pairs):
    remove = {frozenset((a1, a2)) for a1, a2 in pairs}
    topology._bonds = [
        bond for bond in topology._bonds
        if frozenset((bond.atom1, bond.atom2)) not in remove
    ]


def _parse_ssbond_records(pdb_path):
    pairs = []
    try:
        with Path(pdb_path).open("rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("SSBOND"):
                    continue
                c1 = line[15].strip()
                r1 = line[17:21].strip()
                i1 = line[21].strip()
                c2 = line[29].strip()
                r2 = line[31:35].strip()
                i2 = line[35].strip()
                if c1 and r1 and c2 and r2:
                    pairs.append((c1, r1, i1, c2, r2, i2))
    except FileNotFoundError:
        return []
    return pairs


def _find_residue(topology, chain_id, resid, insertion):
    insertion = insertion or ""
    for chain in topology.chains():
        if chain.id != chain_id:
            continue
        for res in chain.residues():
            if res.id == resid and (res.insertionCode or "") == insertion:
                return res
    return None


def _add_disulfides_from_ssbond(modeller, ssbond_pairs):
    added = 0
    for c1, r1, i1, c2, r2, i2 in ssbond_pairs:
        res1 = _find_residue(modeller.topology, c1, r1, i1)
        res2 = _find_residue(modeller.topology, c2, r2, i2)
        if res1 is None or res2 is None:
            continue
        if res1.name != "CYS" or res2.name != "CYS":
            continue
        a1 = next((a for a in res1.atoms() if a.name == "SG"), None)
        a2 = next((a for a in res2.atoms() if a.name == "SG"), None)
        if a1 is None or a2 is None:
            continue
        if any(
            (b.atom1 is a1 and b.atom2 is a2) or (b.atom1 is a2 and b.atom2 is a1)
            for b in modeller.topology.bonds()
        ):
            continue
        modeller.topology.addBond(a1, a2)
        added += 1
    return added


def add_disulfide_bonds(modeller, pdb_path=None, min_nm=0.18, max_nm=0.30):
    sg_atoms = [
        atom for atom in modeller.topology.atoms()
        if atom.name == "SG" and atom.residue.name == "CYS"
    ]

    if pdb_path:
        ssbond_pairs = _parse_ssbond_records(pdb_path)
        if ssbond_pairs:
            _add_disulfides_from_ssbond(modeller, ssbond_pairs)

    # Remove any existing SG-SG bonds that are outside the QC range.
    bad_pairs = []
    for bond in list(modeller.topology.bonds()):
        a1, a2 = bond.atom1, bond.atom2
        if (
            a1.name == "SG"
            and a2.name == "SG"
            and a1.residue.name == "CYS"
            and a2.residue.name == "CYS"
        ):
            d = _distance_nm(modeller.positions, a1, a2)
            if d < min_nm or d > max_nm:
                bad_pairs.append((a1, a2))
    if bad_pairs:
        _remove_bonds(modeller.topology, bad_pairs)

    existing = {frozenset((b.atom1, b.atom2)) for b in modeller.topology.bonds()}

    # Add SG-SG bonds based on distance (QC window).
    candidates = []
    for i, a1 in enumerate(sg_atoms):
        for a2 in sg_atoms[i + 1 :]:
            if a1.residue is a2.residue:
                continue
            d = _distance_nm(modeller.positions, a1, a2)
            if min_nm <= d <= max_nm:
                candidates.append((d, a1, a2))

    bonded = set()
    for d, a1, a2 in sorted(candidates, key=lambda x: x[0]):
        key = frozenset((a1, a2))
        if key in existing or a1 in bonded or a2 in bonded:
            continue
        modeller.topology.addBond(a1, a2)
        existing.add(key)
        bonded.add(a1)
        bonded.add(a2)

    disulfide_residues = set()
    for bond in modeller.topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        if (
            a1.name == "SG"
            and a2.name == "SG"
            and a1.residue.name == "CYS"
            and a2.residue.name == "CYS"
        ):
            disulfide_residues.add(a1.residue)
            disulfide_residues.add(a2.residue)
    return disulfide_residues


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


def cysteine_variants(topology, force_reduced_cys=True):
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
            if residue in disulfide:
                variants.append("CYX")
            else:
                variants.append("CYS" if force_reduced_cys else None)
        else:
            variants.append(None)
    return variants


def solvate(modeller, ph, pdb_path=None, force_reduced_cys=True):
    forcefield = ForceField(*FORCEFIELD_FILES)
    modeller = strip_hydrogens(modeller)
    add_disulfide_bonds(modeller, pdb_path=pdb_path)
    variants = cysteine_variants(modeller.topology, force_reduced_cys=force_reduced_cys)
    modeller.addHydrogens(forcefield, pH=ph, variants=variants)
    modeller.addSolvent(
        forcefield,
        padding=WATER_PADDING_NM * nanometer,
        ionicStrength=IONIC_STRENGTH_M * molar,
    )
    return forcefield, modeller
