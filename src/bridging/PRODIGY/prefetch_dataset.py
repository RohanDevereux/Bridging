from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from bridging.MD.paths import DATA_CSV, resolve_dataset
from bridging.MD.prefetch_dataset import prefetch as prefetch_pdbs_for_dataset
from bridging.MD.prefetch_pdbs import ensure_pdb_cached

from .dataset import ProdigyRequest, parse_request_row
from .paths import default_results_path
from .runner import run_prodigy_delta_g_kcalmol, validate_prodigy_executable


RESULT_COLUMNS = [
    "dataset",
    "row_index",
    "cache_key",
    "pdb_id",
    "ligand_group",
    "receptor_group",
    "temperature_k",
    "temperature_c",
    "delta_g_kcal_mol",
    "status",
    "message",
]


def _temperature_c(temp_k: float | None) -> float | None:
    if temp_k is None or pd.isna(temp_k):
        return None
    return float(temp_k) - 273.15


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
    req: ProdigyRequest,
    delta_g: float | None,
    status: str,
    message: str = "",
) -> dict:
    return {
        "dataset": str(dataset_path),
        "row_index": req.row_index,
        "cache_key": req.cache_key,
        "pdb_id": req.pdb_id,
        "ligand_group": req.ligand_group,
        "receptor_group": req.receptor_group,
        "temperature_k": req.temperature_k,
        "temperature_c": _temperature_c(req.temperature_k),
        "delta_g_kcal_mol": delta_g,
        "status": status,
        "message": message,
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


def fetch_and_save_dataset_estimates(
    dataset_path: str | Path | None = None,
    out_path: str | Path | None = None,
    limit: int | None = None,
    prodigy_exe: str = "prodigy",
    refresh: bool = False,
    prefetch_pdbs: bool = True,
) -> Path:
    prodigy_exe = validate_prodigy_executable(prodigy_exe)
    dataset_path = resolve_dataset(dataset_path, DATA_CSV)
    dataset_path = Path(dataset_path)
    out_path = Path(out_path) if out_path else default_results_path(dataset_path)

    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)
    rows = df.to_dict("records")
    total = len(rows)

    print(f"[PRODIGY] dataset={dataset_path} rows={total} out={out_path}")
    if prefetch_pdbs:
        print("[PRODIGY] prefetching PDBs into local cache")
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
            record = {
                "dataset": str(dataset_path),
                "row_index": row_index,
                "cache_key": "",
                "pdb_id": "",
                "ligand_group": "",
                "receptor_group": "",
                "temperature_k": None,
                "temperature_c": None,
                "delta_g_kcal_mol": None,
                "status": "parse_error",
                "message": str(exc),
            }
            records_by_row[row_index] = record
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
            record = _record_from_request(
                dataset_path=dataset_path,
                req=req,
                delta_g=delta_g,
                status="ok",
                message="reused cached estimate",
            )
            records_by_row[req.row_index] = record
            _save_results(out_path, records_by_row)
            print(
                f"[CACHE] {row_index + 1}/{total} {req.pdb_id} "
                f"lig={req.ligand_group} rec={req.receptor_group} dG={delta_g:.3f}"
            )
            continue

        try:
            pdb_file, _ = ensure_pdb_cached(req.pdb_id)
            delta_g = run_prodigy_delta_g_kcalmol(
                pdb_path=pdb_file,
                ligand_group=req.ligand_group,
                receptor_group=req.receptor_group,
                temperature_k=req.temperature_k,
                prodigy_exe=prodigy_exe,
            )
            key_cache[req.cache_key] = delta_g
            ok_count += 1
            record = _record_from_request(
                dataset_path=dataset_path,
                req=req,
                delta_g=delta_g,
                status="ok",
            )
            records_by_row[req.row_index] = record
            print(
                f"[OK] {row_index + 1}/{total} {req.pdb_id} "
                f"lig={req.ligand_group} rec={req.receptor_group} dG={delta_g:.3f}"
            )
        except Exception as exc:
            fail_count += 1
            message = str(exc)
            record = _record_from_request(
                dataset_path=dataset_path,
                req=req,
                delta_g=None,
                status="fail",
                message=message,
            )
            records_by_row[req.row_index] = record
            print(f"[FAIL] {row_index + 1}/{total} {req.pdb_id}: {message}")
            failure_log = out_path.parent / "prodigy_failures.log"
            failure_log.parent.mkdir(parents=True, exist_ok=True)
            with failure_log.open("a", encoding="utf-8") as f:
                f.write(
                    f"row_index={row_index} pdb={req.pdb_id} cache_key={req.cache_key}\n"
                    f"{traceback.format_exc()}\n"
                )
        _save_results(out_path, records_by_row)

    print(
        f"[PRODIGY] done rows={total} ok={ok_count} reused={cached_count} "
        f"skipped={skip_count} fail={fail_count} out={out_path}"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and save PRODIGY delta G estimates for each row in a dataset."
    )
    parser.add_argument("--dataset", help="CSV path to use (defaults to MD DATA_CSV)")
    parser.add_argument("--out", help="Output CSV path")
    parser.add_argument("--limit", type=int, help="Optional row limit")
    parser.add_argument("--prodigy-exe", default="prodigy", help="PRODIGY executable path")
    parser.add_argument("--refresh", action="store_true", help="Recompute even if cached")
    parser.add_argument(
        "--no-pdb-prefetch",
        action="store_true",
        help="Skip dataset-level PDB prefetch step",
    )
    args = parser.parse_args()

    try:
        fetch_and_save_dataset_estimates(
            dataset_path=args.dataset,
            out_path=args.out,
            limit=args.limit,
            prodigy_exe=args.prodigy_exe,
            refresh=args.refresh,
            prefetch_pdbs=not args.no_pdb_prefetch,
        )
    except Exception as exc:
        raise SystemExit(f"[PRODIGY] {exc}") from exc


if __name__ == "__main__":
    main()
