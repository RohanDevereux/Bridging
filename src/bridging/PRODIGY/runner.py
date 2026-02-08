from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from bridging.MD.paths import PDB_CACHE_DIR
from bridging.MD.prefetch_pdbs import ensure_pdb_cached

from .dataset import make_cache_key


def _k_to_c(temp_k: float | None) -> float | None:
    if temp_k is None or pd.isna(temp_k):
        return None
    return float(temp_k) - 273.15


def _parse_delta_g(stdout: str, stderr: str = "") -> float:
    text = "\n".join([stdout or "", stderr or ""]).strip()
    if not text:
        raise RuntimeError("PRODIGY produced no output.")

    for token in reversed(re.split(r"\s+", text)):
        try:
            return float(token)
        except ValueError:
            continue
    raise RuntimeError(f"Could not parse delta G from PRODIGY output: {text!r}")


def validate_prodigy_executable(prodigy_exe: str = "prodigy") -> str:
    exe = str(prodigy_exe).strip()
    if not exe:
        raise FileNotFoundError("PRODIGY executable is empty.")

    if Path(exe).exists():
        return exe

    resolved = shutil.which(exe)
    if resolved:
        return exe

    raise FileNotFoundError(
        f"Could not find PRODIGY executable '{exe}'. "
        "Install with `pip install prodigy-prot` and ensure it is on PATH, "
        "or pass --prodigy-exe <full-path-to-prodigy>."
    )


def run_prodigy_delta_g_kcalmol(
    pdb_path: Path,
    ligand_group: str,
    receptor_group: str,
    temperature_k: float | None = None,
    prodigy_exe: str = "prodigy",
) -> float:
    prodigy_exe = validate_prodigy_executable(prodigy_exe)
    cmd = [
        prodigy_exe,
        "-q",
        "--selection",
        ligand_group,
        receptor_group,
    ]
    temp_c = _k_to_c(temperature_k)
    if temp_c is not None:
        cmd += ["--temperature", f"{temp_c:.2f}"]
    cmd.append(str(pdb_path))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "PRODIGY failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
            "Install with `pip install prodigy-prot` if needed."
        )
    return _parse_delta_g(proc.stdout, proc.stderr)


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


def get_prodigy_estimate(
    pdb_id: str,
    ligand_group: str,
    receptor_group: str,
    temperature_k: float | None = None,
    cache_path: Path | None = None,
    prefer_prefetched: bool = True,
    prodigy_exe: str = "prodigy",
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

    pdb_file, _ = ensure_pdb_cached(str(pdb_id), cache_dir=PDB_CACHE_DIR)
    return run_prodigy_delta_g_kcalmol(
        pdb_path=pdb_file,
        ligand_group=ligand_group,
        receptor_group=receptor_group,
        temperature_k=temperature_k,
        prodigy_exe=prodigy_exe,
    )
