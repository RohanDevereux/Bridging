from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from openmm.app import PDBFile

from bridging.MD.paths import PDB_CACHE_DIR
from bridging.MD.prefetch_pdbs import ensure_pdb_cached
from bridging.MD.prepare_complex import select_chains
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


def _parse_group(group: str) -> list[str]:
    return [c.upper() for c in parse_chain_group(group)]


def _write_subset_pdb(in_pdb: Path, out_pdb: Path, chain_ids: list[str]):
    pdb = PDBFile(str(in_pdb))
    available = {c.id.upper() for c in pdb.topology.chains()}
    missing = [c for c in chain_ids if c.upper() not in available]
    if missing:
        raise RuntimeError(f"Missing chain(s) in {in_pdb.name}: {missing}; available={sorted(available)}")
    modeller = select_chains(pdb.topology, pdb.positions, chain_ids)
    with out_pdb.open("w", encoding="utf-8") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)


def _write_leap_input(path: Path):
    path.write_text(
        "\n".join(
            [
                "source leaprc.protein.ff14SB",
                "source leaprc.water.tip3p",
                "rec = loadpdb rec.pdb",
                "lig = loadpdb lig.pdb",
                "com = loadpdb com.pdb",
                "saveamberparm rec rec.prmtop rec.inpcrd",
                "saveamberparm lig lig.prmtop lig.inpcrd",
                "saveamberparm com com.prmtop com.inpcrd",
                "quit",
                "",
            ]
        ),
        encoding="utf-8",
    )


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


def _write_mmpbsa_input(path: Path, *, igb: int, saltcon: float, interval: int):
    path.write_text(
        "\n".join(
            [
                "&general",
                f"  interval={int(interval)},",
                "  verbose=1,",
                "/",
                "&gb",
                f"  igb={int(igb)},",
                f"  saltcon={float(saltcon):.3f},",
                "/",
                "",
            ]
        ),
        encoding="utf-8",
    )


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
    igb: int = 5,
    saltcon: float = 0.150,
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
    _write_subset_pdb(Path(pdb_path), work_dir / "rec.pdb", rec_chains)
    _write_subset_pdb(Path(pdb_path), work_dir / "lig.pdb", lig_chains)
    _write_subset_pdb(Path(pdb_path), work_dir / "com.pdb", all_chains)

    _write_leap_input(work_dir / "leap.in")
    _run_checked([tleap_exe, "-f", "leap.in"], cwd=work_dir)

    if trajectory_path is None:
        _write_cpptraj_input(work_dir / "cpptraj.in")
        _run_checked([cpptraj_exe, "-p", "com.prmtop", "-i", "cpptraj.in"], cwd=work_dir)
        traj_path = work_dir / "traj_single.nc"
    else:
        traj_path = Path(trajectory_path).resolve()

    _write_mmpbsa_input(work_dir / "mmpbsa.in", igb=igb, saltcon=saltcon, interval=interval)
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
    igb: int = 5,
    saltcon: float = 0.150,
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
        igb=igb,
        saltcon=saltcon,
        interval=interval,
    )
