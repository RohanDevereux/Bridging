from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

from bridging.MD.paths import PDB_CACHE_DIR
from bridging.MD.prefetch_pdbs import ensure_pdb_cached

from .config import (
    D0_NM,
    DIST_CLIP_NM,
    DTYPE,
    INTERFACE_K_FRAMES,
    K_NM,
    N_INTERFACE,
    USE_LOG_DIST,
)
from .contact_map import contact_distance_channels
from .interface import select_interface_atoms


def _parse_chain_group(value) -> list[str]:
    if value is None:
        return []
    tokens = re.findall(r"[A-Za-z0-9]", str(value))
    return list(dict.fromkeys(tokens))


def _row_chain_groups(row: dict) -> tuple[list[str], list[str]]:
    normalized = {str(k).strip().lower(): k for k in row.keys()}
    if "chains_1" in normalized and "chains_2" in normalized:
        return _parse_chain_group(row[normalized["chains_1"]]), _parse_chain_group(row[normalized["chains_2"]])
    if "ligand chains" in normalized and "receptor chains" in normalized:
        return _parse_chain_group(row[normalized["ligand chains"]]), _parse_chain_group(row[normalized["receptor chains"]])
    if "ligand_chains" in normalized and "receptor_chains" in normalized:
        return _parse_chain_group(row[normalized["ligand_chains"]]), _parse_chain_group(row[normalized["receptor_chains"]])
    if "complex_pdb" in normalized:
        chains = str(row[normalized["complex_pdb"]]).split("_", 1)[-1]
        left, right = chains.split(":")
        return _parse_chain_group(left), _parse_chain_group(right)
    return [], []


def _row_pdb_id(row: dict) -> str | None:
    normalized = {str(k).strip().lower(): k for k in row.keys()}
    for key in ("pdb", "pdb_id"):
        if key in normalized:
            value = str(row[normalized[key]]).strip().upper()
            if re.fullmatch(r"[A-Z0-9]{4}", value):
                return value
    if "complex_pdb" in normalized:
        value = str(row[normalized["complex_pdb"]]).strip()
        if "_" in value:
            value = value.split("_", 1)[0]
        value = value.upper()
        if re.fullmatch(r"[A-Z0-9]{4}", value):
            return value
    return None


def _default_out_dir(dataset_path: Path) -> Path:
    return dataset_path.parent.parent / "generatedData" / "featurization_smoke" / dataset_path.stem


def run_smoke(
    *,
    dataset_path: str | Path,
    out_dir: str | Path | None = None,
    pdb_root: str | Path = PDB_CACHE_DIR,
    fetch_missing: bool = False,
    limit: int | None = None,
) -> dict:
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir) if out_dir else _default_out_dir(dataset_path)
    pdb_root = Path(pdb_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(int(limit)).copy()
    records = df.to_dict("records")

    report_rows = []
    ok = 0
    fail = 0

    total = len(records)
    print(f"[SMOKE] dataset={dataset_path} rows={total} out={out_dir}")

    for row_index, row in enumerate(records):
        pdb_id = _row_pdb_id(row)
        chains_1, chains_2 = _row_chain_groups(row)

        if pdb_id is None:
            fail += 1
            report_rows.append(
                {
                    "row_index": row_index,
                    "pdb_id": None,
                    "chains_1": ",".join(chains_1),
                    "chains_2": ",".join(chains_2),
                    "status": "fail",
                    "message": "could not parse PDB id",
                    "output_path": "",
                    "shape": "",
                }
            )
            print(f"[FAIL] {row_index + 1}/{total} pdb=? could not parse PDB id")
            continue

        try:
            if fetch_missing:
                pdb_path, _ = ensure_pdb_cached(pdb_id, cache_dir=pdb_root)
            else:
                pdb_path = pdb_root / f"{pdb_id}.pdb"
                if not pdb_path.exists():
                    raise FileNotFoundError(f"PDB not cached: {pdb_path}")

            traj = md.load_pdb(str(pdb_path))
            idx1, idx2 = select_interface_atoms(
                traj,
                chains_1,
                chains_2,
                N_INTERFACE,
                method="stable",
                k_frames=INTERFACE_K_FRAMES,
                d0_nm=D0_NM,
                k_nm=K_NM,
            )
            X = contact_distance_channels(
                traj,
                idx1,
                idx2,
                stride=1,
                d0_nm=D0_NM,
                k_nm=K_NM,
                d_clip_nm=DIST_CLIP_NM,
                use_log_dist=USE_LOG_DIST,
                dtype=DTYPE,
            )
            X = X[:1].astype(np.float16, copy=False)

            out_file = out_dir / f"{row_index:04d}_{pdb_id}_first_frame.npy"
            np.save(out_file, X)

            meta = {
                "row_index": row_index,
                "pdb_id": pdb_id,
                "chains_1": chains_1,
                "chains_2": chains_2,
                "output_file": str(out_file),
                "shape": list(X.shape),
            }
            (out_dir / f"{row_index:04d}_{pdb_id}_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

            ok += 1
            report_rows.append(
                {
                    "row_index": row_index,
                    "pdb_id": pdb_id,
                    "chains_1": ",".join(chains_1),
                    "chains_2": ",".join(chains_2),
                    "status": "ok",
                    "message": "",
                    "output_path": str(out_file),
                    "shape": "x".join(str(v) for v in X.shape),
                }
            )
            print(f"[OK] {row_index + 1}/{total} {pdb_id} shape={tuple(X.shape)}")
        except Exception as exc:
            fail += 1
            report_rows.append(
                {
                    "row_index": row_index,
                    "pdb_id": pdb_id,
                    "chains_1": ",".join(chains_1),
                    "chains_2": ",".join(chains_2),
                    "status": "fail",
                    "message": str(exc),
                    "output_path": "",
                    "shape": "",
                }
            )
            print(f"[FAIL] {row_index + 1}/{total} {pdb_id}: {exc}")

    report_df = pd.DataFrame(report_rows)
    report_path = out_dir / "smoke_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"[SMOKE] done rows={total} ok={ok} fail={fail} report={report_path}")
    return {
        "rows": total,
        "ok": ok,
        "fail": fail,
        "report_csv": report_path,
        "out_dir": out_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test featurization from PDB only (no MD): "
            "generate first-frame 64x64 contact+distance features per dataset row."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset CSV")
    parser.add_argument("--out-dir", help="Output directory for first-frame features")
    parser.add_argument("--pdb-root", default=str(PDB_CACHE_DIR), help="PDB cache directory")
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Download missing PDBs from RCSB if not present in cache",
    )
    parser.add_argument("--limit", type=int, help="Optional row limit")
    args = parser.parse_args()

    run_smoke(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        pdb_root=args.pdb_root,
        fetch_missing=args.fetch_missing,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
