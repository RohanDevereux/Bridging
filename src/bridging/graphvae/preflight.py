from __future__ import annotations

import argparse
import glob
import inspect
import json
import random
import shutil
from pathlib import Path

import h5py
import mdtraj as md
import pandas as pd

from .ids import canonical_complex_id, sanitize_filename_token


def _check_deeprank2(expected_version: str) -> str:
    import deeprank2  # noqa: PLC0415
    from deeprank2.query import ProteinProteinInterfaceQuery, QueryCollection  # noqa: PLC0415

    version = str(getattr(deeprank2, "__version__", "unknown"))
    if version != expected_version:
        raise RuntimeError(
            f"deeprank2 version mismatch: expected={expected_version} found={version}. "
            "Reinstall environment with pip install -e ."
        )

    q_sig = inspect.signature(ProteinProteinInterfaceQuery)
    p_sig = inspect.signature(QueryCollection.process)
    if "influence_radius" not in q_sig.parameters:
        raise RuntimeError("ProteinProteinInterfaceQuery is missing 'influence_radius' parameter.")
    if "influence_radius" in p_sig.parameters:
        raise RuntimeError("QueryCollection.process unexpectedly has 'influence_radius' parameter.")
    return version


def _check_records(dataset_csv: Path) -> int:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise RuntimeError(f"Dataset CSV has no rows: {dataset_csv}")
    return int(len(df))


def _check_msms() -> str:
    msms = shutil.which("msms")
    if msms is None:
        raise RuntimeError(
            "msms executable not found on PATH. DeepRank exposure features (res_depth/hse) require msms. "
            "Load/install msms before running graphvae prepare."
        )
    return msms


def _check_freesasa() -> str:
    import freesasa  # noqa: PLC0415

    return str(getattr(freesasa, "__version__", "unknown"))


def _check_md_root(md_root: Path, sample_n: int, require_done: int) -> dict:
    if not md_root.exists():
        raise FileNotFoundError(f"MD root does not exist: {md_root}")

    dirs = sorted([d for d in md_root.iterdir() if d.is_dir()])
    done_dirs = [d for d in dirs if (d / "DONE").exists()]
    if len(done_dirs) < require_done:
        raise RuntimeError(
            f"Not enough DONE complexes under {md_root}: found={len(done_dirs)} require>={require_done}"
        )

    rng = random.Random(2026)
    picks = done_dirs if len(done_dirs) <= sample_n else rng.sample(done_dirs, sample_n)
    checked = 0
    for d in picks:
        traj = d / "traj_full.nc"
        top = d / "topology_full.pdb"
        if not traj.exists() or not top.exists():
            raise RuntimeError(f"{d.name}: missing traj_full.nc or topology_full.pdb")
        md.load(str(traj), top=str(top), frame=0)
        checked += 1

    return {
        "n_complex_dirs": len(dirs),
        "n_done_dirs": len(done_dirs),
        "sample_checked": checked,
    }


def _as_path_list(values: list[str] | None) -> list[Path]:
    if not values:
        return []
    out: list[Path] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if "*" in text or "?" in text or "[" in text:
            out.extend(Path(p) for p in glob.glob(text))
        else:
            out.append(Path(text))
    unique = []
    seen = set()
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _check_deeprank_overlap(dataset_csv: Path, md_root: Path, hdf5_paths: list[Path]) -> dict:
    if not hdf5_paths:
        raise RuntimeError("No HDF5 paths were provided for DeepRank overlap check.")
    for p in hdf5_paths:
        if not p.exists():
            raise FileNotFoundError(f"DeepRank HDF5 not found: {p}")

    df = pd.read_csv(dataset_csv)
    done_models = set()
    for row in df.to_dict("records"):
        complex_id = canonical_complex_id(row)
        if complex_id is None:
            continue
        pdb_id = str(complex_id).split("__", 1)[0]
        if (md_root / pdb_id / "DONE").exists():
            done_models.add(sanitize_filename_token(complex_id))

    hdf5_models = set()
    for p in hdf5_paths:
        with h5py.File(p, "r") as h5:
            for key in h5.keys():
                hdf5_models.add(str(key).split(":")[-1].strip())

    overlap = done_models.intersection(hdf5_models)
    report = {
        "n_hdf5_files": int(len(hdf5_paths)),
        "n_hdf5_models": int(len(hdf5_models)),
        "n_done_models": int(len(done_models)),
        "n_overlap": int(len(overlap)),
        "missing_sample": sorted(done_models - hdf5_models)[:20],
    }
    if report["n_overlap"] < 1:
        raise RuntimeError(
            "DeepRank HDF5 has zero overlap with DONE complexes after model-id normalization. "
            f"diagnostic={json.dumps(report)}"
        )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-fast preflight checks for graphvae post-MD pipeline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--md-root", required=True)
    parser.add_argument("--expected-deeprank2-version", default="3.1.0")
    parser.add_argument("--sample-done", type=int, default=20)
    parser.add_argument("--require-done", type=int, default=50)
    parser.add_argument("--deep-rank-hdf5", nargs="*", help="Optional DeepRank HDF5 paths (or glob patterns).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    deeprank2_version = _check_deeprank2(str(args.expected_deeprank2_version))
    msms_path = _check_msms()
    freesasa_version = _check_freesasa()
    n_rows = _check_records(Path(args.dataset))
    md_stats = _check_md_root(
        Path(args.md_root),
        sample_n=int(args.sample_done),
        require_done=int(args.require_done),
    )
    overlap_stats = None
    hdf5_paths = _as_path_list(args.deep_rank_hdf5)
    if hdf5_paths:
        overlap_stats = _check_deeprank_overlap(Path(args.dataset), Path(args.md_root), hdf5_paths)
        print(
            "[PREFLIGHT] deeprank_overlap "
            f"hdf5_files={overlap_stats['n_hdf5_files']} "
            f"hdf5_models={overlap_stats['n_hdf5_models']} "
            f"done_models={overlap_stats['n_done_models']} "
            f"overlap={overlap_stats['n_overlap']}"
        )
    print(
        f"[PREFLIGHT] ok deeprank2={deeprank2_version} freesasa={freesasa_version} "
        f"msms={msms_path} dataset_rows={n_rows} "
        f"complex_dirs={md_stats['n_complex_dirs']} done={md_stats['n_done_dirs']} "
        f"sample_checked={md_stats['sample_checked']}"
    )


if __name__ == "__main__":
    main()
