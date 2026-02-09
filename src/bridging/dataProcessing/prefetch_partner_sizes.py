from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .filter_ppb_dataset import (
    RCSB_OK,
    _is_valid_pdb_id,
    _load_ca_coords_by_chain,
    _normalize_subgroup,
    _parse_chain_group,
    _size_key,
    partner_lengths_from_chain_groups_rcsb,
)


def _partner_modeled_residues(
    coords_by_chain: dict[str, np.ndarray],
    ligand_chains: str,
    receptor_chains: str,
) -> tuple[int, int]:
    lig = _parse_chain_group(ligand_chains)
    rec = _parse_chain_group(receptor_chains)
    n_lig = int(sum(int(coords_by_chain.get(ch, np.zeros((0, 3))).shape[0]) for ch in lig))
    n_rec = int(sum(int(coords_by_chain.get(ch, np.zeros((0, 3))).shape[0]) for ch in rec))
    return n_lig, n_rec


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Prefetch partner size metadata for dataset rows and save to CSV for future filtering."
        )
    )
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--pdb-root", required=True, help="Directory containing <PDB>.pdb files")
    ap.add_argument("--limit", type=int, help="Optional row limit for debugging")
    ap.add_argument("--use-rcsb", action="store_true", help="Also fetch sequence lengths via RCSB API")
    ap.add_argument("--rcsb-cache-json", default="rcsb_lengths_cache.json")
    ap.add_argument("--col-pdb", default="PDB")
    ap.add_argument("--col-ligand", default="Ligand Chains")
    ap.add_argument("--col-receptor", default="Receptor Chains")
    ap.add_argument("--col-subgroup", default="Subgroup")
    args = ap.parse_args()

    if args.use_rcsb and not RCSB_OK:
        raise RuntimeError("`--use-rcsb` requested but py-rcsb-api is not installed.")

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    pdb_root = Path(args.pdb_root)

    df = pd.read_csv(in_csv)
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()

    rcsb_cache = {}
    cache_path = Path(args.rcsb_cache_json)
    if cache_path.exists():
        try:
            rcsb_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            rcsb_cache = {}

    seen_keys = set()
    pdb_coords_cache: dict[str, dict[str, np.ndarray]] = {}
    rows = []

    total = len(df)
    for i, row in enumerate(df.to_dict("records"), start=1):
        if i % 500 == 0 or i == total:
            print(f"[SIZES] {i}/{total}")

        pdb_id = str(row.get(args.col_pdb, "")).strip().upper()
        ligand = str(row.get(args.col_ligand, ""))
        receptor = str(row.get(args.col_receptor, ""))
        subgroup = _normalize_subgroup(row.get(args.col_subgroup, None))

        if not _is_valid_pdb_id(pdb_id):
            continue

        key = _size_key(pdb_id, ligand, receptor)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        pdb_path = pdb_root / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            rows.append(
                {
                    "size_key": key,
                    "pdb_id": pdb_id,
                    "ligand_chains": ",".join(_parse_chain_group(ligand)),
                    "receptor_chains": ",".join(_parse_chain_group(receptor)),
                    "subgroup": subgroup,
                    "pdb_path": str(pdb_path),
                    "status": "missing_pdb",
                }
            )
            continue

        if pdb_id not in pdb_coords_cache:
            pdb_coords_cache[pdb_id] = _load_ca_coords_by_chain(pdb_path)
        coords = pdb_coords_cache[pdb_id]
        lig_modeled, rec_modeled = _partner_modeled_residues(coords, ligand, receptor)

        lig_seq = None
        rec_seq = None
        rcsb_status = "skipped"
        if args.use_rcsb:
            try:
                lens = partner_lengths_from_chain_groups_rcsb(
                    pdb_id, ligand, receptor, rcsb_cache
                )
                lig_seq = lens.get("ligand_total_residues", None)
                rec_seq = lens.get("receptor_total_residues", None)
                rcsb_status = "ok"
            except Exception as exc:
                rcsb_status = f"error:{type(exc).__name__}"

        rows.append(
            {
                "size_key": key,
                "pdb_id": pdb_id,
                "ligand_chains": ",".join(_parse_chain_group(ligand)),
                "receptor_chains": ",".join(_parse_chain_group(receptor)),
                "subgroup": subgroup,
                "pdb_path": str(pdb_path),
                "ligand_modeled_residues": lig_modeled,
                "receptor_modeled_residues": rec_modeled,
                "ligand_seq_len": lig_seq,
                "receptor_seq_len": rec_seq,
                "rcsb_status": rcsb_status,
                "status": "ok",
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    if args.use_rcsb:
        try:
            cache_path.write_text(json.dumps(rcsb_cache, indent=2), encoding="utf-8")
        except Exception:
            pass

    ok = int((out_df["status"] == "ok").sum()) if not out_df.empty and "status" in out_df.columns else 0
    missing = int((out_df["status"] == "missing_pdb").sum()) if not out_df.empty and "status" in out_df.columns else 0
    print(
        f"[DONE] size_rows={len(out_df)} ok={ok} missing_pdb={missing} "
        f"out={out_csv}"
    )


if __name__ == "__main__":
    main()
