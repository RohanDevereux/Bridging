from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from bridging.MD.paths import DATA_CSV, resolve_dataset
from bridging.MD.prefetch_dataset import prefetch as prefetch_pdbs_for_dataset
from bridging.MD.prefetch_pdbs import ensure_pdb_cached

from .dataset import MMGBSARequest, parse_request_row
from .paths import MMGBSA_OUT_DIR, default_results_path
from .runner import run_mmgbsa_delta_g_kcalmol, validate_mmgbsa_tools


RESULT_COLUMNS = [
    "dataset",
    "row_index",
    "cache_key",
    "pdb_id",
    "ligand_group",
    "receptor_group",
    "temperature_k",
    "delta_g_kcal_mol",
    "status",
    "message",
    "source_pdb",
    "source_kind",
    "work_dir",
]


def _load_existing(out_path: Path, dataset_path: Path) -> dict[int, dict]:
    if not out_path.exists():
        return {}

    df = pd.read_csv(out_path)
    if "dataset" in df.columns:
        same_dataset = df["dataset"] == str(dataset_path)
        if same_dataset.any():
            df = df[same_dataset]
    if "row_index" not in df.columns:
        return {}

    existing = {}
    for row in df.to_dict("records"):
        try:
            row_idx = int(row["row_index"])
        except Exception:
            continue
        existing[row_idx] = row
    return existing


def _build_success_cache(records_by_row: dict[int, dict]) -> dict[str, float]:
    cache = {}
    for record in records_by_row.values():
        if record.get("status") != "ok":
            continue
        key = record.get("cache_key")
        if not key:
            continue
        value = pd.to_numeric(pd.Series([record.get("delta_g_kcal_mol")]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        cache[str(key)] = float(value)
    return cache


def _record_from_request(
    dataset_path: Path,
    req: MMGBSARequest,
    *,
    delta_g: float | None,
    status: str,
    message: str = "",
    source_pdb: Path | None = None,
    source_kind: str = "",
    work_dir: Path | None = None,
) -> dict:
    return {
        "dataset": str(dataset_path),
        "row_index": req.row_index,
        "cache_key": req.cache_key,
        "pdb_id": req.pdb_id,
        "ligand_group": req.ligand_group,
        "receptor_group": req.receptor_group,
        "temperature_k": req.temperature_k,
        "delta_g_kcal_mol": delta_g,
        "status": status,
        "message": message,
        "source_pdb": str(source_pdb) if source_pdb else "",
        "source_kind": source_kind,
        "work_dir": str(work_dir) if work_dir else "",
    }


def _save_results(out_path: Path, records_by_row: dict[int, dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records_by_row:
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(out_path, index=False)
        return

    rows = [records_by_row[k] for k in sorted(records_by_row.keys())]
    df = pd.DataFrame(rows)
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[RESULT_COLUMNS]
    df.to_csv(out_path, index=False)


def _resolve_source(req: MMGBSARequest, md_root: Path | None) -> tuple[Path, str, Path | None]:
    if md_root is not None:
        md_top = md_root / req.pdb_id / "topology_protein.pdb"
        if md_top.exists():
            for traj_name in ("traj_protein.nc", "traj_protein.dcd"):
                traj = md_root / req.pdb_id / traj_name
                if traj.exists():
                    return md_top, "md_top_with_traj", traj
            return md_top, "md_top_single", None

        md_final = md_root / req.pdb_id / "final.pdb"
        if md_final.exists():
            return md_final, "md_final_single", None
    cached, _ = ensure_pdb_cached(req.pdb_id)
    return cached, "rcsb_cache", None


def _cache_key_to_dir(cache_key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in cache_key)


def fetch_and_save_dataset_estimates(
    dataset_path: str | Path | None = None,
    out_path: str | Path | None = None,
    limit: int | None = None,
    refresh: bool = False,
    prefetch_pdbs: bool = True,
    md_root: str | Path | None = None,
    work_root: str | Path | None = None,
    tleap_exe: str = "tleap",
    cpptraj_exe: str = "cpptraj",
    mmpbsa_exe: str = "MMPBSA.py",
    igb: int = 5,
    saltcon: float = 0.150,
    interval: int = 1,
) -> Path:
    validate_mmgbsa_tools(tleap_exe=tleap_exe, cpptraj_exe=cpptraj_exe, mmpbsa_exe=mmpbsa_exe)
    dataset_path = resolve_dataset(dataset_path, DATA_CSV)
    dataset_path = Path(dataset_path)
    out_path = Path(out_path) if out_path else default_results_path(dataset_path)
    md_root_path = Path(md_root) if md_root else None
    work_root_path = Path(work_root) if work_root else MMGBSA_OUT_DIR / "_work" / dataset_path.stem

    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)
    rows = df.to_dict("records")
    total = len(rows)

    print(f"[MMGBSA] dataset={dataset_path} rows={total} out={out_path}")
    if prefetch_pdbs:
        print("[MMGBSA] prefetching PDBs into local cache")
        prefetch_pdbs_for_dataset(dataset_path, limit=limit)

    records_by_row = _load_existing(out_path, dataset_path)
    key_cache = _build_success_cache(records_by_row)

    ok_count = 0
    fail_count = 0
    skip_count = 0
    cached_count = 0

    for row_index, row in enumerate(rows):
        try:
            req = parse_request_row(row, row_index=row_index)
        except Exception as exc:
            fail_count += 1
            print(f"[FAIL] {row_index + 1}/{total} parse error: {exc}")
            records_by_row[row_index] = {
                "dataset": str(dataset_path),
                "row_index": row_index,
                "cache_key": "",
                "pdb_id": "",
                "ligand_group": "",
                "receptor_group": "",
                "temperature_k": None,
                "delta_g_kcal_mol": None,
                "status": "parse_error",
                "message": str(exc),
                "source_pdb": "",
                "source_kind": "",
                "work_dir": "",
            }
            _save_results(out_path, records_by_row)
            continue

        existing = records_by_row.get(req.row_index)
        if (
            not refresh
            and existing is not None
            and str(existing.get("cache_key", "")) == req.cache_key
            and existing.get("status") == "ok"
        ):
            skip_count += 1
            print(f"[SKIP] {row_index + 1}/{total} {req.pdb_id} already saved")
            continue

        if not refresh and req.cache_key in key_cache:
            delta_g = key_cache[req.cache_key]
            cached_count += 1
            records_by_row[req.row_index] = _record_from_request(
                dataset_path,
                req,
                delta_g=delta_g,
                status="ok",
                message="reused cached estimate",
            )
            _save_results(out_path, records_by_row)
            print(
                f"[CACHE] {row_index + 1}/{total} {req.pdb_id} "
                f"lig={req.ligand_group} rec={req.receptor_group} dG={delta_g:.3f}"
            )
            continue

        source_pdb = None
        source_kind = ""
        work_dir = work_root_path / _cache_key_to_dir(req.cache_key)
        try:
            source_pdb, source_kind, traj_path = _resolve_source(req, md_root_path)
            delta_g = run_mmgbsa_delta_g_kcalmol(
                pdb_path=source_pdb,
                ligand_group=req.ligand_group,
                receptor_group=req.receptor_group,
                work_dir=work_dir,
                trajectory_path=traj_path,
                tleap_exe=tleap_exe,
                cpptraj_exe=cpptraj_exe,
                mmpbsa_exe=mmpbsa_exe,
                igb=igb,
                saltcon=saltcon,
                interval=interval,
            )
            key_cache[req.cache_key] = delta_g
            ok_count += 1
            records_by_row[req.row_index] = _record_from_request(
                dataset_path,
                req,
                delta_g=delta_g,
                status="ok",
                source_pdb=source_pdb,
                source_kind=source_kind,
                work_dir=work_dir,
            )
            print(
                f"[OK] {row_index + 1}/{total} {req.pdb_id} source={source_kind} "
                f"lig={req.ligand_group} rec={req.receptor_group} dG={delta_g:.3f}"
            )
        except Exception as exc:
            fail_count += 1
            message = str(exc)
            records_by_row[req.row_index] = _record_from_request(
                dataset_path,
                req,
                delta_g=None,
                status="fail",
                message=message,
                source_pdb=source_pdb,
                source_kind=source_kind,
                work_dir=work_dir,
            )
            print(f"[FAIL] {row_index + 1}/{total} {req.pdb_id}: {message}")
            failure_log = out_path.parent / "mmgbsa_failures.log"
            failure_log.parent.mkdir(parents=True, exist_ok=True)
            with failure_log.open("a", encoding="utf-8") as f:
                f.write(
                    f"row_index={row_index} pdb={req.pdb_id} cache_key={req.cache_key}\n"
                    f"{traceback.format_exc()}\n"
                )
        _save_results(out_path, records_by_row)

    print(
        f"[MMGBSA] done rows={total} ok={ok_count} reused={cached_count} "
        f"skipped={skip_count} fail={fail_count} out={out_path}"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and save MM/GBSA delta G estimates for each row in a dataset."
    )
    parser.add_argument("--dataset", help="CSV path to use (defaults to MD DATA_CSV)")
    parser.add_argument("--out", help="Output CSV path")
    parser.add_argument("--limit", type=int, help="Optional row limit")
    parser.add_argument("--refresh", action="store_true", help="Recompute even if cached")
    parser.add_argument(
        "--no-pdb-prefetch",
        action="store_true",
        help="Skip dataset-level PDB prefetch step",
    )
    parser.add_argument(
        "--md-root",
        help=(
            "Optional MD root. Uses <md_root>/<PDB>/topology_protein.pdb + traj_protein.nc "
            "when available, else falls back to final.pdb."
        ),
    )
    parser.add_argument("--work-root", help="Directory for MMGBSA working files")
    parser.add_argument("--tleap-exe", default="tleap")
    parser.add_argument("--cpptraj-exe", default="cpptraj")
    parser.add_argument("--mmpbsa-exe", default="MMPBSA.py")
    parser.add_argument("--igb", type=int, default=5)
    parser.add_argument("--saltcon", type=float, default=0.150)
    parser.add_argument("--interval", type=int, default=1)
    args = parser.parse_args()

    try:
        fetch_and_save_dataset_estimates(
            dataset_path=args.dataset,
            out_path=args.out,
            limit=args.limit,
            refresh=args.refresh,
            prefetch_pdbs=not args.no_pdb_prefetch,
            md_root=args.md_root,
            work_root=args.work_root,
            tleap_exe=args.tleap_exe,
            cpptraj_exe=args.cpptraj_exe,
            mmpbsa_exe=args.mmpbsa_exe,
            igb=args.igb,
            saltcon=args.saltcon,
            interval=args.interval,
        )
    except Exception as exc:
        raise SystemExit(f"[MMGBSA] {exc}") from exc


if __name__ == "__main__":
    main()
