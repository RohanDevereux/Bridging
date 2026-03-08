from __future__ import annotations

import argparse
import inspect
import random
from pathlib import Path

import mdtraj as md
import pandas as pd


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-fast preflight checks for graphvae post-MD pipeline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--md-root", required=True)
    parser.add_argument("--expected-deeprank2-version", default="3.1.0")
    parser.add_argument("--sample-done", type=int, default=20)
    parser.add_argument("--require-done", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    deeprank2_version = _check_deeprank2(str(args.expected_deeprank2_version))
    n_rows = _check_records(Path(args.dataset))
    md_stats = _check_md_root(
        Path(args.md_root),
        sample_n=int(args.sample_done),
        require_done=int(args.require_done),
    )
    print(
        f"[PREFLIGHT] ok deeprank2={deeprank2_version} dataset_rows={n_rows} "
        f"complex_dirs={md_stats['n_complex_dirs']} done={md_stats['n_done_dirs']} "
        f"sample_checked={md_stats['sample_checked']}"
    )


if __name__ == "__main__":
    main()
