from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from bridging.MD.paths import DATA_CSV, resolve_dataset
from bridging.MD.prefetch_dataset import prefetch as prefetch_pdbs_for_dataset
from bridging.MD.prefetch_pdbs import ensure_pdb_cached
from bridging.utils.dataset_rows import parse_chain_group
from bridging.graphvae.chain_remap import build_raw_to_md_chain_map

from .dataset import MMGBSARequest, parse_request_row
from .paths import MMGBSA_OUT_DIR, default_results_path
from .runner import pdb_has_chains, run_mmgbsa_delta_g_kcalmol, validate_mmgbsa_tools


RESULT_COLUMNS = [
    "dataset",
    "row_index",
    "cache_key",
    "pdb_id",
    "ligand_group",
    "receptor_group",
    "used_ligand_group",
    "used_receptor_group",
    "temperature_k",
    "solvation_model",
    "igb",
    "saltcon",
    "istrng",
    "start_frame",
    "end_frame",
    "interval",
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


def _config_signature(
    *,
    solvation_model: str,
    igb: int,
    saltcon: float,
    istrng: float,
    start_frame: int,
    end_frame: int | None,
    interval: int,
) -> tuple:
    return (
        str(solvation_model).strip().lower(),
        int(igb),
        round(float(saltcon), 6),
        round(float(istrng), 6),
        int(start_frame),
        (None if end_frame is None else int(end_frame)),
        int(interval),
    )


def _record_signature(record: dict) -> tuple:
    return _config_signature(
        solvation_model=str(record.get("solvation_model", "gb")),
        igb=int(record.get("igb", 5) or 5),
        saltcon=float(record.get("saltcon", 0.150) or 0.150),
        istrng=float(record.get("istrng", 0.150) or 0.150),
        start_frame=int(record.get("start_frame", 1) or 1),
        end_frame=(
            None
            if record.get("end_frame") in (None, "", "None") or pd.isna(record.get("end_frame"))
            else int(float(record.get("end_frame")))
        ),
        interval=int(record.get("interval", 1) or 1),
    )


def _build_success_cache(records_by_row: dict[int, dict]) -> dict[tuple[str, tuple], float]:
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
        cache[(str(key), _record_signature(record))] = float(value)
    return cache


def _record_from_request(
    dataset_ref: Path,
    req: MMGBSARequest,
    *,
    solvation_model: str,
    igb: int,
    saltcon: float,
    istrng: float,
    start_frame: int,
    end_frame: int | None,
    interval: int,
    delta_g: float | None,
    status: str,
    message: str = "",
    used_ligand_group: str | None = None,
    used_receptor_group: str | None = None,
    source_pdb: Path | None = None,
    source_kind: str = "",
    work_dir: Path | None = None,
) -> dict:
    return {
        "dataset": str(dataset_ref),
        "row_index": req.row_index,
        "cache_key": req.cache_key,
        "pdb_id": req.pdb_id,
        "ligand_group": req.ligand_group,
        "receptor_group": req.receptor_group,
        "used_ligand_group": used_ligand_group or req.ligand_group,
        "used_receptor_group": used_receptor_group or req.receptor_group,
        "temperature_k": req.temperature_k,
        "solvation_model": str(solvation_model).strip().lower(),
        "igb": int(igb),
        "saltcon": float(saltcon),
        "istrng": float(istrng),
        "start_frame": int(start_frame),
        "end_frame": (None if end_frame is None else int(end_frame)),
        "interval": int(interval),
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


def _format_group(chain_ids: list[str]) -> str:
    return ",".join([str(c).strip().upper() for c in chain_ids if str(c).strip()])


def _resolve_source(
    req: MMGBSARequest,
    md_root: Path | None,
    *,
    chain_map_cache: dict[str, tuple[dict[str, str], list[str], dict]] | None = None,
) -> tuple[Path, str, Path | None, str, str]:
    raw_lig = [c.upper() for c in parse_chain_group(req.ligand_group)]
    raw_rec = [c.upper() for c in parse_chain_group(req.receptor_group)]
    needed_chains = [c for c in (raw_lig + raw_rec) if c.strip()]
    if md_root is not None:
        md_top = md_root / req.pdb_id / "topology_protein.pdb"
        if md_top.exists():
            used_lig = raw_lig
            used_rec = raw_rec
            source_kind = "md_top"

            raw_pdb, _ = ensure_pdb_cached(req.pdb_id)
            if raw_pdb.exists():
                cache_key = str(req.pdb_id).strip().upper()
                if chain_map_cache is not None and cache_key in chain_map_cache:
                    chain_map, md_chain_order, _ = chain_map_cache[cache_key]
                else:
                    chain_map, md_chain_order, report = build_raw_to_md_chain_map(raw_pdb, md_top)
                    if chain_map_cache is not None:
                        chain_map_cache[cache_key] = (chain_map, md_chain_order, report)
                remapped_lig = [chain_map.get(c, c) for c in raw_lig]
                remapped_rec = [chain_map.get(c, c) for c in raw_rec]
                remapped_needed = [c for c in (remapped_lig + remapped_rec) if c.strip()]
                if remapped_needed and pdb_has_chains(md_top, remapped_needed):
                    used_lig = remapped_lig
                    used_rec = remapped_rec
                    source_kind = "md_top_remapped"
                    needed_chains = remapped_needed

            if pdb_has_chains(md_top, needed_chains):
                used_ligand_group = _format_group(used_lig)
                used_receptor_group = _format_group(used_rec)
                for traj_name in ("traj_protein.nc", "traj_protein.dcd"):
                    traj = md_root / req.pdb_id / traj_name
                    if traj.exists():
                        return md_top, f"{source_kind}_with_traj", traj, used_ligand_group, used_receptor_group
                return md_top, f"{source_kind}_single", None, used_ligand_group, used_receptor_group

        md_final = md_root / req.pdb_id / "final.pdb"
        if md_final.exists() and pdb_has_chains(md_final, needed_chains):
            return md_final, "md_final_single", None, req.ligand_group, req.receptor_group

    cached, _ = ensure_pdb_cached(req.pdb_id)
    if not pdb_has_chains(cached, needed_chains):
        raise RuntimeError(
            f"No available source PDB contains required chains {sorted(set(needed_chains))} for {req.pdb_id}."
        )
    return cached, "rcsb_cache", None, req.ligand_group, req.receptor_group


def _cache_key_to_dir(cache_key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in cache_key)


def fetch_and_save_dataset_estimates(
    dataset_path: str | Path | None = None,
    dataset_label: str | Path | None = None,
    out_path: str | Path | None = None,
    limit: int | None = None,
    refresh: bool = False,
    prefetch_pdbs: bool = True,
    md_root: str | Path | None = None,
    work_root: str | Path | None = None,
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
) -> Path:
    validate_mmgbsa_tools(tleap_exe=tleap_exe, cpptraj_exe=cpptraj_exe, mmpbsa_exe=mmpbsa_exe)
    dataset_path = resolve_dataset(dataset_path, DATA_CSV)
    dataset_path = Path(dataset_path)
    out_path = Path(out_path) if out_path else default_results_path(dataset_path)
    dataset_ref = Path(dataset_label) if dataset_label else dataset_path
    md_root_path = Path(md_root) if md_root else None
    work_root_path = Path(work_root) if work_root else MMGBSA_OUT_DIR / "_work" / dataset_path.stem

    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)
    rows = df.to_dict("records")
    total = len(rows)

    print(
        f"[MMGBSA] dataset={dataset_path} dataset_label={dataset_ref} rows={total} out={out_path} "
        f"model={str(solvation_model).lower()} start={int(start_frame)} "
        f"end={('NA' if end_frame is None else int(end_frame))} interval={int(interval)}"
    )
    if prefetch_pdbs:
        print("[MMGBSA] prefetching PDBs into local cache")
        prefetch_pdbs_for_dataset(dataset_path, limit=limit)

    records_by_row = _load_existing(out_path, dataset_ref)
    key_cache = _build_success_cache(records_by_row)
    chain_map_cache: dict[str, tuple[dict[str, str], list[str], dict]] = {}
    target_sig = _config_signature(
        solvation_model=solvation_model,
        igb=igb,
        saltcon=saltcon,
        istrng=istrng,
        start_frame=start_frame,
        end_frame=end_frame,
        interval=interval,
    )

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
                "dataset": str(dataset_ref),
                "row_index": row_index,
                "cache_key": "",
                "pdb_id": "",
                "ligand_group": "",
                "receptor_group": "",
                "used_ligand_group": "",
                "used_receptor_group": "",
                "temperature_k": None,
                "solvation_model": str(solvation_model).strip().lower(),
                "igb": int(igb),
                "saltcon": float(saltcon),
                "istrng": float(istrng),
                "start_frame": int(start_frame),
                "end_frame": (None if end_frame is None else int(end_frame)),
                "interval": int(interval),
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
            and _record_signature(existing) == target_sig
        ):
            skip_count += 1
            print(f"[SKIP] {row_index + 1}/{total} {req.pdb_id} already saved")
            continue

        cache_lookup = (req.cache_key, target_sig)
        if not refresh and cache_lookup in key_cache:
            delta_g = key_cache[cache_lookup]
            cached_count += 1
            records_by_row[req.row_index] = _record_from_request(
                dataset_ref,
                req,
                solvation_model=solvation_model,
                igb=igb,
                saltcon=saltcon,
                istrng=istrng,
                start_frame=start_frame,
                end_frame=end_frame,
                interval=interval,
                delta_g=delta_g,
                status="ok",
                message="reused cached estimate",
                used_ligand_group=req.ligand_group,
                used_receptor_group=req.receptor_group,
            )
            _save_results(out_path, records_by_row)
            print(
                f"[CACHE] {row_index + 1}/{total} {req.pdb_id} "
                f"lig={req.ligand_group} rec={req.receptor_group} dG={delta_g:.3f}"
            )
            continue

        source_pdb = None
        source_kind = ""
        used_ligand_group = req.ligand_group
        used_receptor_group = req.receptor_group
        work_dir = work_root_path / _cache_key_to_dir(req.cache_key)
        try:
            source_pdb, source_kind, traj_path, used_ligand_group, used_receptor_group = _resolve_source(
                req,
                md_root_path,
                chain_map_cache=chain_map_cache,
            )
            delta_g = run_mmgbsa_delta_g_kcalmol(
                pdb_path=source_pdb,
                ligand_group=used_ligand_group,
                receptor_group=used_receptor_group,
                work_dir=work_dir,
                trajectory_path=traj_path,
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
            key_cache[cache_lookup] = delta_g
            ok_count += 1
            records_by_row[req.row_index] = _record_from_request(
                dataset_ref,
                req,
                solvation_model=solvation_model,
                igb=igb,
                saltcon=saltcon,
                istrng=istrng,
                start_frame=start_frame,
                end_frame=end_frame,
                interval=interval,
                delta_g=delta_g,
                status="ok",
                used_ligand_group=used_ligand_group,
                used_receptor_group=used_receptor_group,
                source_pdb=source_pdb,
                source_kind=source_kind,
                work_dir=work_dir,
            )
            print(
                f"[OK] {row_index + 1}/{total} {req.pdb_id} source={source_kind} "
                f"lig={used_ligand_group} rec={used_receptor_group} dG={delta_g:.3f}"
            )
        except Exception as exc:
            fail_count += 1
            message = str(exc)
            records_by_row[req.row_index] = _record_from_request(
                dataset_ref,
                req,
                solvation_model=solvation_model,
                igb=igb,
                saltcon=saltcon,
                istrng=istrng,
                start_frame=start_frame,
                end_frame=end_frame,
                interval=interval,
                delta_g=None,
                status="fail",
                message=message,
                used_ligand_group=used_ligand_group,
                used_receptor_group=used_receptor_group,
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
    parser.add_argument("--dataset-label", help="Canonical dataset path to store in results (useful when processing shards).")
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
    parser.add_argument("--solvation-model", choices=["gb", "pb"], default="gb")
    parser.add_argument("--igb", type=int, default=5)
    parser.add_argument("--saltcon", type=float, default=0.150)
    parser.add_argument("--istrng", type=float, default=0.150)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--interval", type=int, default=1)
    args = parser.parse_args()

    try:
        fetch_and_save_dataset_estimates(
            dataset_path=args.dataset,
            dataset_label=args.dataset_label,
            out_path=args.out,
            limit=args.limit,
            refresh=args.refresh,
            prefetch_pdbs=not args.no_pdb_prefetch,
            md_root=args.md_root,
            work_root=args.work_root,
            tleap_exe=args.tleap_exe,
            cpptraj_exe=args.cpptraj_exe,
            mmpbsa_exe=args.mmpbsa_exe,
            solvation_model=args.solvation_model,
            igb=args.igb,
            saltcon=args.saltcon,
            istrng=args.istrng,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            interval=args.interval,
        )
    except Exception as exc:
        raise SystemExit(f"[MMGBSA] {exc}") from exc


if __name__ == "__main__":
    main()
