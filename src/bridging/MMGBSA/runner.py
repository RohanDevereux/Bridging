from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from openmm import unit
from openmm.app import PDBFile
from openmm.app import Topology

from bridging.MD.paths import PDB_CACHE_DIR
from bridging.MD.prefetch_pdbs import ensure_pdb_cached
from bridging.MD.prepare_complex import (
    add_disulfide_bonds,
    drop_non_protein_residues,
    select_chains,
    strip_hydrogens,
)
from bridging.utils.dataset_rows import parse_chain_group

from .dataset import make_cache_key


def _run_checked(cmd: list[str], cwd: Path | None = None):
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )
    return proc


def available_chain_ids(pdb_path: Path) -> list[str]:
    pdb = PDBFile(str(pdb_path))
    return sorted({str(c.id).strip().upper() for c in pdb.topology.chains()})


def pdb_has_chains(pdb_path: Path, chain_ids: list[str]) -> bool:
    available = set(available_chain_ids(pdb_path))
    wanted = {str(c).strip().upper() for c in chain_ids}
    return wanted.issubset(available)


def _parse_group(group: str) -> list[str]:
    return [c.upper() for c in parse_chain_group(group)]


def _copy_tleap_safe_subset(
    *,
    topology,
    positions,
    out_pdb: Path,
    disulfide_residues: set,
) -> list[tuple[int, int]]:
    top_new = Topology()
    atom_map = {}
    residue_order = {}
    pos_new = []
    residue_counter = 0

    for chain in topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        new_chain = top_new.addChain(chain.id)
        for residue in residues:
            residue_counter += 1
            residue_order[residue] = residue_counter
            residue_name = residue.name
            if residue_name == "CYS" and residue in disulfide_residues:
                residue_name = "CYX"
            new_residue = top_new.addResidue(
                residue_name,
                new_chain,
                residue.id,
                getattr(residue, "insertionCode", ""),
            )
            for atom in residue.atoms():
                new_atom = top_new.addAtom(atom.name, atom.element, new_residue, atom.id)
                atom_map[atom] = new_atom
                pos_new.append(positions[atom.index])

    disulfide_pairs: set[tuple[int, int]] = set()
    for bond in topology.bonds():
        atom1 = atom_map.get(bond.atom1)
        atom2 = atom_map.get(bond.atom2)
        if atom1 is None or atom2 is None:
            continue
        top_new.addBond(atom1, atom2)
        if (
            bond.atom1.name == "SG"
            and bond.atom2.name == "SG"
            and bond.atom1.residue in residue_order
            and bond.atom2.residue in residue_order
        ):
            pair = tuple(sorted((residue_order[bond.atom1.residue], residue_order[bond.atom2.residue])))
            disulfide_pairs.add(pair)

    if hasattr(positions, "unit"):
        pos_unit = positions.unit
        pos_new = unit.Quantity([p.value_in_unit(pos_unit) for p in pos_new], pos_unit)
    else:
        pos_new = unit.Quantity(pos_new, unit.nanometer)

    with out_pdb.open("w", encoding="utf-8") as f:
        PDBFile.writeFile(top_new, pos_new, f)
    return sorted(disulfide_pairs)


def _write_subset_pdb(in_pdb: Path, out_pdb: Path, chain_ids: list[str]) -> list[tuple[int, int]]:
    pdb = PDBFile(str(in_pdb))
    available = {c.id.upper() for c in pdb.topology.chains()}
    missing = [c for c in chain_ids if c.upper() not in available]
    if missing:
        raise RuntimeError(f"Missing chain(s) in {in_pdb.name}: {missing}; available={sorted(available)}")
    modeller = select_chains(pdb.topology, pdb.positions, chain_ids)
    modeller = drop_non_protein_residues(modeller)
    modeller = strip_hydrogens(modeller)
    disulfide_residues = add_disulfide_bonds(modeller, pdb_path=in_pdb)
    return _copy_tleap_safe_subset(
        topology=modeller.topology,
        positions=modeller.positions,
        out_pdb=out_pdb,
        disulfide_residues=disulfide_residues,
    )


def _append_bond_lines(lines: list[str], unit_name: str, residue_pairs: list[tuple[int, int]]) -> None:
    for res1, res2 in residue_pairs:
        lines.append(f"bond {unit_name}.{int(res1)}.SG {unit_name}.{int(res2)}.SG")


def _write_leap_input(
    path: Path,
    *,
    rec_disulfides: list[tuple[int, int]],
    lig_disulfides: list[tuple[int, int]],
    com_disulfides: list[tuple[int, int]],
):
    lines = [
        "source leaprc.protein.ff14SB",
        "source leaprc.water.tip3p",
        "rec = loadpdb rec.pdb",
        "lig = loadpdb lig.pdb",
        "com = loadpdb com.pdb",
    ]
    _append_bond_lines(lines, "rec", rec_disulfides)
    _append_bond_lines(lines, "lig", lig_disulfides)
    _append_bond_lines(lines, "com", com_disulfides)
    lines.extend(
        [
            "saveamberparm rec rec.prmtop rec.inpcrd",
            "saveamberparm lig lig.prmtop lig.inpcrd",
            "saveamberparm com com.prmtop com.inpcrd",
            "quit",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_cpptraj_input(path: Path):
    path.write_text(
        "\n".join(
            [
                "trajin com.inpcrd",
                "trajout traj_single.nc netcdf",
                "run",
                "quit",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_mmpbsa_input(
    path: Path,
    *,
    solvation_model: str,
    igb: int,
    saltcon: float,
    istrng: float,
    start_frame: int,
    end_frame: int | None,
    interval: int,
):
    model = str(solvation_model).strip().lower()
    if model not in {"gb", "pb"}:
        raise ValueError(f"Unsupported solvation_model={solvation_model!r}; expected 'gb' or 'pb'.")

    lines = [
        "&general",
        f"  startframe={int(start_frame)},",
        f"  interval={int(interval)},",
        "  verbose=1,",
    ]
    if end_frame is not None and int(end_frame) >= int(start_frame):
        lines.append(f"  endframe={int(end_frame)},")
    lines.extend(["/"])

    if model == "gb":
        lines.extend(
            [
                "&gb",
                f"  igb={int(igb)},",
                f"  saltcon={float(saltcon):.3f},",
                "/",
            ]
        )
    else:
        lines.extend(
            [
                "&pb",
                f"  istrng={float(istrng):.3f},",
                "/",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_final_results_delta_g(final_results: Path) -> float:
    if not final_results.exists():
        raise RuntimeError(f"Missing MMGBSA output: {final_results}")
    text = final_results.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "DELTA TOTAL" not in line.upper():
            continue
        vals = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", line)
        if vals:
            return float(vals[0])
    m = re.search(
        r"DELTA\s+G\s+binding\s*=\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(1))
    raise RuntimeError(f"Could not parse DELTA TOTAL from {final_results}")


def validate_mmgbsa_tools(
    *,
    tleap_exe: str = "tleap",
    cpptraj_exe: str = "cpptraj",
    mmpbsa_exe: str = "MMPBSA.py",
) -> tuple[str, str, str]:
    out = []
    for exe in (tleap_exe, cpptraj_exe, mmpbsa_exe):
        text = str(exe).strip()
        if not text:
            raise FileNotFoundError("MMGBSA executable path is empty.")
        if Path(text).exists() or shutil.which(text):
            out.append(text)
            continue
        raise FileNotFoundError(
            f"Could not find executable: {text}. "
            "Install AmberTools (tleap/cpptraj/MMPBSA.py) and ensure they are on PATH."
        )
    return tuple(out)


def run_mmgbsa_delta_g_kcalmol(
    *,
    pdb_path: Path,
    ligand_group: str,
    receptor_group: str,
    work_dir: Path,
    trajectory_path: Path | None = None,
    tleap_exe: str = "tleap",
    cpptraj_exe: str = "cpptraj",
    mmpbsa_exe: str = "MMPBSA.py",
    solvation_model: str = "gb",
    igb: int = 5,
    saltcon: float = 0.150,
    istrng: float = 0.150,
    start_frame: int = 1,
    end_frame: int | None = None,
    interval: int = 1,
) -> float:
    tleap_exe, cpptraj_exe, mmpbsa_exe = validate_mmgbsa_tools(
        tleap_exe=tleap_exe, cpptraj_exe=cpptraj_exe, mmpbsa_exe=mmpbsa_exe
    )

    pdb_path = Path(pdb_path).resolve()
    lig_chains = _parse_group(ligand_group)
    rec_chains = _parse_group(receptor_group)
    all_chains = list(dict.fromkeys(lig_chains + rec_chains))
    if not lig_chains or not rec_chains:
        raise RuntimeError("Empty ligand/receptor chain groups.")

    work_dir.mkdir(parents=True, exist_ok=True)
    rec_disulfides = _write_subset_pdb(Path(pdb_path), work_dir / "rec.pdb", rec_chains)
    lig_disulfides = _write_subset_pdb(Path(pdb_path), work_dir / "lig.pdb", lig_chains)
    com_disulfides = _write_subset_pdb(Path(pdb_path), work_dir / "com.pdb", all_chains)

    _write_leap_input(
        work_dir / "leap.in",
        rec_disulfides=rec_disulfides,
        lig_disulfides=lig_disulfides,
        com_disulfides=com_disulfides,
    )
    _run_checked([tleap_exe, "-f", "leap.in"], cwd=work_dir)

    if trajectory_path is None:
        _write_cpptraj_input(work_dir / "cpptraj.in")
        _run_checked([cpptraj_exe, "-p", "com.prmtop", "-i", "cpptraj.in"], cwd=work_dir)
        traj_path = work_dir / "traj_single.nc"
    else:
        traj_path = Path(trajectory_path).resolve()

    _write_mmpbsa_input(
        work_dir / "mmpbsa.in",
        solvation_model=solvation_model,
        igb=igb,
        saltcon=saltcon,
        istrng=istrng,
        start_frame=max(1, int(start_frame)),
        end_frame=(None if end_frame is None else int(end_frame)),
        interval=max(1, int(interval)),
    )
    _run_checked(
        [
            mmpbsa_exe,
            "-O",
            "-i",
            "mmpbsa.in",
            "-cp",
            "com.prmtop",
            "-rp",
            "rec.prmtop",
            "-lp",
            "lig.prmtop",
            "-y",
            str(traj_path),
            "-o",
            "FINAL_RESULTS_MMPBSA.dat",
        ],
        cwd=work_dir,
    )
    return _parse_final_results_delta_g(work_dir / "FINAL_RESULTS_MMPBSA.dat")


def get_prefetched_estimate(
    cache_path: Path,
    pdb_id: str,
    ligand_group: str,
    receptor_group: str,
    temperature_k: float | None = None,
) -> float | None:
    if not cache_path.exists():
        return None

    cache_key = make_cache_key(pdb_id, ligand_group, receptor_group, temperature_k)
    df = pd.read_csv(cache_path)
    if "cache_key" not in df.columns or "delta_g_kcal_mol" not in df.columns:
        return None

    matched = df[df["cache_key"] == cache_key]
    if matched.empty:
        return None
    series = pd.to_numeric(matched["delta_g_kcal_mol"], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _cache_key_to_dir(cache_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_key)


def get_mmgbsa_estimate(
    *,
    pdb_id: str,
    ligand_group: str,
    receptor_group: str,
    temperature_k: float | None = None,
    cache_path: Path | None = None,
    work_root: Path | None = None,
    prefer_prefetched: bool = True,
    source_pdb_path: Path | None = None,
    trajectory_path: Path | None = None,
    tleap_exe: str = "tleap",
    cpptraj_exe: str = "cpptraj",
    mmpbsa_exe: str = "MMPBSA.py",
    solvation_model: str = "gb",
    igb: int = 5,
    saltcon: float = 0.150,
    istrng: float = 0.150,
    start_frame: int = 1,
    end_frame: int | None = None,
    interval: int = 1,
) -> float:
    if prefer_prefetched and cache_path is not None:
        cached = get_prefetched_estimate(
            cache_path=cache_path,
            pdb_id=pdb_id,
            ligand_group=ligand_group,
            receptor_group=receptor_group,
            temperature_k=temperature_k,
        )
        if cached is not None:
            return cached

    pdb_path = Path(source_pdb_path) if source_pdb_path else ensure_pdb_cached(str(pdb_id), cache_dir=PDB_CACHE_DIR)[0]
    cache_key = make_cache_key(pdb_id, ligand_group, receptor_group, temperature_k)
    root = Path(work_root) if work_root else Path(".") / "mmgbsa_work"
    work_dir = root / _cache_key_to_dir(cache_key)

    return run_mmgbsa_delta_g_kcalmol(
        pdb_path=pdb_path,
        ligand_group=ligand_group,
        receptor_group=receptor_group,
        work_dir=work_dir,
        trajectory_path=trajectory_path,
        tleap_exe=tleap_exe,
        cpptraj_exe=cpptraj_exe,
        mmpbsa_exe=mmpbsa_exe,
        solvation_model=solvation_model,
        igb=igb,
        saltcon=saltcon,
        istrng=istrng,
        start_frame=start_frame,
        end_frame=end_frame,
        interval=interval,
    )
