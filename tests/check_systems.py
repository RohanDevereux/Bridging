import argparse
import traceback
from pathlib import Path

import pandas as pd

from bridging.MD.paths import DATA_CSV, PPB_DATA_CSV, PDB_CACHE_DIR, resolve_dataset
from bridging.MD.prepare_complex import (
    load_and_fix,
    select_chains,
    drop_non_protein_residues,
    mark_disulfides,
    cysteine_variants,
    solvate,
)
from bridging.MD.simulate import build_system


def _parse_chain_group(value):
    if pd.isna(value):
        return []
    cleaned = "".join([c for c in str(value) if c.isalnum()])
    return list(dict.fromkeys(list(cleaned)))


def _chain_ids(row):
    normalized = {str(k).strip().lower(): k for k in row.keys()}
    if "chains_1" in normalized and "chains_2" in normalized:
        c1 = _parse_chain_group(row[normalized["chains_1"]])
        c2 = _parse_chain_group(row[normalized["chains_2"]])
        return list(dict.fromkeys(c1 + c2))
    if "ligand chains" in normalized and "receptor chains" in normalized:
        c1 = _parse_chain_group(row[normalized["ligand chains"]])
        c2 = _parse_chain_group(row[normalized["receptor chains"]])
        return list(dict.fromkeys(c1 + c2))
    if "ligand_chains" in normalized and "receptor_chains" in normalized:
        c1 = _parse_chain_group(row[normalized["ligand_chains"]])
        c2 = _parse_chain_group(row[normalized["receptor_chains"]])
        return list(dict.fromkeys(c1 + c2))
    if "complex_pdb" in normalized:
        left = str(row["complex_pdb"]).split("_", 1)[-1]
        left, right = left.split(":")
        return list(dict.fromkeys(list(left) + list(right)))
    raise ValueError("No chain columns found (expected Chains_1/Chains_2 or complex_pdb).")


def run_checks(dataset_path, limit=None):
    dataset_path = resolve_dataset(dataset_path, DATA_CSV)
    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)

    records = df.to_dict("records")
    total = len(records)
    print(f"[CHECK] dataset={dataset_path} rows={total}")

    ok = 0
    fail = 0
    for idx, row in enumerate(records, start=1):
        pdb_id = str(row.get("PDB") or row.get("PDB_ID") or "").upper()
        if not pdb_id:
            print(f"[SKIP] {idx}/{total} missing PDB")
            continue

        print(f"[CHECK] {idx}/{total} {pdb_id}")
        try:
            pdb_file = PDB_CACHE_DIR / f"{pdb_id}.pdb"
            fixer = load_and_fix(pdb_file)
            chain_ids = _chain_ids(row)
            modeller = select_chains(fixer.topology, fixer.positions, chain_ids)
            modeller = drop_non_protein_residues(modeller)
            modeller = mark_disulfides(modeller)
            variants = cysteine_variants(modeller.topology)
            forcefield, modeller = solvate(modeller, ph=7.0, variants=variants)
            build_system(forcefield, modeller)
            print(f"[OK] {pdb_id}")
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {pdb_id}: {exc}")
            print(traceback.format_exc())
            fail += 1

    print(f"[DONE] ok={ok} fail={fail} total={total}")


def main():
    parser = argparse.ArgumentParser(description="Check OpenMM system formation without running dynamics.")
    parser.add_argument("--dataset", help="CSV path to use instead of default")
    parser.add_argument("--limit", type=int, help="Optional row limit")
    parser.add_argument(
        "--ppb",
        action="store_true",
        help="Use PPB_Affinity_TCR_pMHC.csv as default dataset",
    )
    args = parser.parse_args()

    default_path = PPB_DATA_CSV if args.ppb else DATA_CSV
    run_checks(args.dataset or default_path, limit=args.limit)


if __name__ == "__main__":
    main()
