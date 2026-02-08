import argparse
import json
import re
import traceback
from pathlib import Path

import pandas as pd

from .paths import GENERATED_DIR, PDB_CACHE_DIR
from .prepare_complex import (
    load_and_fix,
    select_chains,
    drop_non_protein_residues,
    solvate,
    SkipComplex,
)
from .simulate import run_simulation


def _pdb_from_complex(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if "_" in s:
        s = s.split("_", 1)[0]
    m = re.match(r"^[A-Za-z0-9]{4}$", s)
    return s.upper() if m else None


def _parse_chain_group(value):
    if pd.isna(value):
        return []
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(value))
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
        left = str(row[normalized["complex_pdb"]]).split("_", 1)[-1]
        left, right = left.split(":")
        return list(dict.fromkeys(list(left) + list(right)))
    raise ValueError("No chain columns found (expected Chains_1/Chains_2, Ligand/Receptor Chains, or complex_pdb).")


def _chain_groups(row):
    normalized = {str(k).strip().lower(): k for k in row.keys()}
    if "chains_1" in normalized and "chains_2" in normalized:
        return str(row[normalized["chains_1"]]), str(row[normalized["chains_2"]])
    if "ligand chains" in normalized and "receptor chains" in normalized:
        return str(row[normalized["ligand chains"]]), str(row[normalized["receptor chains"]])
    if "ligand_chains" in normalized and "receptor_chains" in normalized:
        return str(row[normalized["ligand_chains"]]), str(row[normalized["receptor_chains"]])
    if "complex_pdb" in normalized:
        chains = str(row[normalized["complex_pdb"]]).split("_", 1)[-1]
        left, right = chains.split(":")
        return left, right
    return None, None


def _get_pdb_id(row):
    if "PDB" in row:
        return str(row["PDB"]).upper()
    if "PDB_ID" in row:
        return str(row["PDB_ID"]).upper()
    if "complex_pdb" in row:
        return _pdb_from_complex(row["complex_pdb"])
    return None


def _get_temp_k(row, default=300.0):
    for key in ("Temp_K", "Temperature_K", "Temperature (K)"):
        if key in row:
            return float(row[key])
    return float(default)


def _get_ph(row, default=7.0):
    if "pH" in row:
        try:
            value = float(row["pH"])
            if value == value:
                return value
        except Exception:
            pass
    if "PH" in row:
        try:
            value = float(row["PH"])
            if value == value:
                return value
        except Exception:
            pass
    return float(default)


def run_all(dataset_path, out_dir=None, limit=None):
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)

    out_root = Path(out_dir) if out_dir else GENERATED_DIR / "MD_datasets" / dataset_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    records = df.to_dict("records")
    total = len(records)
    print(f"[RUN] dataset={dataset_path} rows={total} out={out_root}")

    for idx, row in enumerate(records, start=1):
        pdb_id = _get_pdb_id(row)
        if not pdb_id:
            print(f"[SKIP] {idx}/{total} missing PDB")
            continue

        chain_ids = _chain_ids(row)
        chains_1, chains_2 = _chain_groups(row)
        temp_k = _get_temp_k(row)
        ph = _get_ph(row)

        out_dir = out_root / pdb_id
        done_file = out_dir / "DONE"
        if done_file.exists():
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {idx}/{total} {pdb_id}")

        meta = {
            "pdb_id": pdb_id,
            "chain_ids": chain_ids,
            "chains_1": chains_1,
            "chains_2": chains_2,
            "temp_k": temp_k,
            "pH": ph,
            "dataset": str(dataset_path),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        try:
            pdb_file = PDB_CACHE_DIR / f"{pdb_id}.pdb"
            fixer = load_and_fix(pdb_file)
            modeller = select_chains(fixer.topology, fixer.positions, chain_ids)
            modeller = drop_non_protein_residues(modeller)
            forcefield, modeller = solvate(modeller, ph, pdb_path=pdb_file)
            run_simulation(forcefield, modeller, out_dir, temp_k)
            done_file.write_text("ok\n", encoding="utf-8")
            print(f"[OK] {pdb_id}")
        except SkipComplex as exc:
            (out_dir / "skip.log").write_text(str(exc) + "\n", encoding="utf-8")
            print(f"[SKIP] {pdb_id}: {exc}")
        except Exception as exc:
            (out_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[FAIL] {pdb_id}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Run MD for a dataset with flexible column mapping.")
    parser.add_argument("--dataset", required=True, help="CSV path to use")
    parser.add_argument("--out", help="Output root directory")
    parser.add_argument("--limit", type=int, help="Optional row limit")
    args = parser.parse_args()

    run_all(args.dataset, out_dir=args.out, limit=args.limit)


if __name__ == "__main__":
    main()
