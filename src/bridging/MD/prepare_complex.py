from pathlib import Path

import numpy as np
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


class SkipComplex(Exception):
    """Raised when a complex is not suitable for the automated, faithful pipeline."""


def load_and_fix(
    pdb_source,
    *,
    replace_nonstandard=True,
    fail_on_missing_residues=False,
    fail_on_nonstandard=False,
):
    pdb_path = Path(pdb_source)
    if pdb_path.exists():
        fixer = PDBFixer(filename=str(pdb_path))
    else:
        pdb_id = pdb_path.stem if pdb_path.suffix else str(pdb_source)
        fixer = PDBFixer(pdbid=str(pdb_id))

    fixer.findMissingResidues()
    missing_count = sum(len(v) for v in fixer.missingResidues.values())
    if missing_count and fail_on_missing_residues:
        raise SkipComplex(f"Missing residues detected ({missing_count}); skipping.")
    # Prevent missing residues from being inserted.
    fixer.missingResidues = {}

    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues and fail_on_nonstandard and not replace_nonstandard:
        first = fixer.nonstandardResidues[0][0].name
        raise SkipComplex(f"Nonstandard residue found ({first}); skipping.")
    if replace_nonstandard:
        fixer.replaceNonstandardResidues()

    fixer.removeHeterogens(keepWater=False)
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


def parse_remark_465_missing_residues(pdb_path):
    out = []
    with Path(pdb_path).open("rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("REMARK 465"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            res = parts[2].upper()
            if len(res) != 3:
                continue
            chain = parts[3]
            resid = parts[4]
            num = "".join([c for c in resid if c.isdigit() or c == "-"])
            ins = resid[len(num):] if num else ""
            if not num:
                continue
            out.append((res, chain, num, ins))
    return out


def _find_residue(topology, chain_id, resid, insertion):
    insertion = insertion or ""
    for chain in topology.chains():
        if chain.id != chain_id:
            continue
        for res in chain.residues():
            if res.id == resid and (res.insertionCode or "") == insertion:
                return res
    return None


def _resid_int(resid):
    try:
        return int(resid)
    except Exception:
        return None


def _add_disulfides_from_ssbond(modeller, ssbond_pairs, min_nm=0.162, max_nm=0.249):
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
            raise SkipComplex("SSBOND references CYS without SG atom; skipping.")
        d = _distance_nm(modeller.positions, a1, a2)
        if d < min_nm or d > max_nm:
            raise SkipComplex(f"SSBOND SG-SG distance {d:.3f} nm out of QC window; skipping.")
        if any(
            (b.atom1 is a1 and b.atom2 is a2) or (b.atom1 is a2 and b.atom2 is a1)
            for b in modeller.topology.bonds()
        ):
            continue
        modeller.topology.addBond(a1, a2)
        added += 1
    return added


def add_disulfide_bonds(modeller, pdb_path=None, min_nm=0.162, max_nm=0.249):
    if pdb_path:
        ssbond_pairs = _parse_ssbond_records(pdb_path)
        if ssbond_pairs:
            _add_disulfides_from_ssbond(modeller, ssbond_pairs, min_nm=min_nm, max_nm=max_nm)

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


def find_internal_chain_breaks(modeller):
    bonded = set()
    for bond in modeller.topology.bonds():
        bonded.add((bond.atom1, bond.atom2))
        bonded.add((bond.atom2, bond.atom1))

    breaks = {}
    for chain in modeller.topology.chains():
        residues = list(chain.residues())
        for left, right in zip(residues, residues[1:]):
            c_atom = next((a for a in left.atoms() if a.name == "C"), None)
            n_atom = next((a for a in right.atoms() if a.name == "N"), None)
            if c_atom is None or n_atom is None:
                continue
            if (c_atom, n_atom) not in bonded:
                breaks.setdefault(chain.id, []).append((left.id, right.id))
    return breaks


def compute_interface_residues(modeller, cutoff_nm=0.5):
    atoms = [
        atom for atom in modeller.topology.atoms()
        if atom.element is not None and atom.element != hydrogen
    ]
    if not atoms:
        return set()
    pos = np.array(
        [
            [
                modeller.positions[a.index].value_in_unit(unit.nanometer).x,
                modeller.positions[a.index].value_in_unit(unit.nanometer).y,
                modeller.positions[a.index].value_in_unit(unit.nanometer).z,
            ]
            for a in atoms
        ],
        dtype=float,
    )
    chain_ids = [a.residue.chain.id for a in atoms]
    residues = [a.residue for a in atoms]

    cutoff2 = cutoff_nm * cutoff_nm
    interface = set()
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if chain_ids[i] == chain_ids[j]:
                continue
            d = pos[i] - pos[j]
            if float(d.dot(d)) <= cutoff2:
                interface.add(residues[i])
                interface.add(residues[j])
    return interface


def permissive_qc_or_skip(
    pdb_path,
    modeller,
    *,
    max_internal_breaks=3,
    max_internal_missing_total=30,
    max_internal_run=12,
    skip_breaks_near_interface=True,
    interface_cutoff_nm=0.5,
):
    if pdb_path is None:
        return
    remark_missing = parse_remark_465_missing_residues(pdb_path)
    breaks = find_internal_chain_breaks(modeller)

    chain_minmax = {}
    for chain in modeller.topology.chains():
        vals = [_resid_int(res.id) for res in chain.residues()]
        vals = [v for v in vals if v is not None]
        if vals:
            chain_minmax[chain.id] = (min(vals), max(vals))

    internal_missing_by_chain = {}
    for _res, ch, resid, _ins in remark_missing:
        ri = _resid_int(resid)
        if ri is None or ch not in chain_minmax:
            continue
        mn, mx = chain_minmax[ch]
        if mn < ri < mx:
            internal_missing_by_chain.setdefault(ch, []).append(ri)

    internal_total = sum(len(v) for v in internal_missing_by_chain.values())
    max_run = 0
    for _ch, lst in internal_missing_by_chain.items():
        s = sorted(set(lst))
        run = 1
        for a, b in zip(s, s[1:]):
            if b == a + 1:
                run += 1
            else:
                max_run = max(max_run, run)
                run = 1
        max_run = max(max_run, run)

    internal_break_count = sum(len(v) for v in breaks.values())

    if internal_break_count > max_internal_breaks:
        raise SkipComplex(f"Too many internal chain breaks ({internal_break_count}).")
    if internal_total > max_internal_missing_total:
        raise SkipComplex(f"Too many internal missing residues ({internal_total}).")
    if max_run > max_internal_run:
        raise SkipComplex(f"Internal missing run too long ({max_run}).")

    if skip_breaks_near_interface:
        interface = compute_interface_residues(modeller, cutoff_nm=interface_cutoff_nm)
        for chain_id, brs in breaks.items():
            chain = next((c for c in modeller.topology.chains() if c.id == chain_id), None)
            if chain is None:
                continue
            by_id = {r.id: r for r in chain.residues()}
            for left_id, right_id in brs:
                if (by_id.get(left_id) in interface) or (by_id.get(right_id) in interface):
                    raise SkipComplex(f"Chain break near interface: {chain_id}:{left_id}-{right_id}")


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


def solvate(
    modeller,
    ph,
    pdb_path=None,
    force_reduced_cys=True,
    qc_max_internal_breaks=3,
    qc_max_internal_missing_total=30,
    qc_max_internal_run=12,
    qc_skip_breaks_near_interface=True,
    qc_interface_cutoff_nm=0.5,
):
    forcefield = ForceField(*FORCEFIELD_FILES)
    modeller = strip_hydrogens(modeller)
    permissive_qc_or_skip(
        pdb_path,
        modeller,
        max_internal_breaks=qc_max_internal_breaks,
        max_internal_missing_total=qc_max_internal_missing_total,
        max_internal_run=qc_max_internal_run,
        skip_breaks_near_interface=qc_skip_breaks_near_interface,
        interface_cutoff_nm=qc_interface_cutoff_nm,
    )
    add_disulfide_bonds(modeller, pdb_path=pdb_path)
    variants = cysteine_variants(modeller.topology, force_reduced_cys=force_reduced_cys)
    modeller.addHydrogens(forcefield, pH=ph, variants=variants)
    modeller.addSolvent(
        forcefield,
        padding=WATER_PADDING_NM * nanometer,
        ionicStrength=IONIC_STRENGTH_M * molar,
    )
    return forcefield, modeller
