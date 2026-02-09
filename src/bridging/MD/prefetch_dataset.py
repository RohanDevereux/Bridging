import argparse
import re

import pandas as pd
import requests

from .paths import DATA_CSV, PPB_DATA_CSV, PDB_CACHE_DIR, resolve_dataset
from .prefetch_pdbs import ensure_pdb_cached


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


def _collect_pdb_ids(df):
    invalid_samples = []
    invalid_count = 0

    def _coerce_pdb(value):
        nonlocal invalid_count
        if pd.isna(value):
            return None
        s = str(value).strip()
        if not s:
            return None
        if "_" in s:
            s = s.split("_", 1)[0]
        s = s.upper()
        if re.fullmatch(r"[A-Z0-9]{4}", s):
            return s
        invalid_count += 1
        if len(invalid_samples) < 20:
            invalid_samples.append(s)
        return None

    if "PDB" in df.columns:
        series = df["PDB"].apply(_coerce_pdb)
    elif "PDB_ID" in df.columns:
        series = df["PDB_ID"].apply(_coerce_pdb)
    elif "complex_pdb" in df.columns:
        series = df["complex_pdb"].apply(_coerce_pdb)
    else:
        raise ValueError("No PDB column found (expected PDB, PDB_ID, or complex_pdb).")

    pdb_ids = pd.Series(series).dropna().unique().tolist()
    return sorted(pdb_ids), invalid_count, invalid_samples


def prefetch(dataset_path, limit=None):
    dataset_path = resolve_dataset(dataset_path, DATA_CSV)
    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)

    pdb_ids, invalid_count, invalid_samples = _collect_pdb_ids(df)
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[PREFETCH] dataset={dataset_path} count={len(pdb_ids)} "
        f"invalid_pdb_values={invalid_count}"
    )
    if invalid_samples:
        print(f"[PREFETCH] skipped invalid PDB values (sample): {invalid_samples}")

    ok = 0
    fail = 0
    for pdb_id in pdb_ids:
        try:
            _, downloaded = ensure_pdb_cached(pdb_id)
            if downloaded:
                print(f"[OK] {pdb_id}")
            ok += 1
        except requests.HTTPError as exc:
            fail += 1
            status = getattr(exc.response, "status_code", "unknown")
            print(f"[MISS] {pdb_id} http={status}")
        except Exception as exc:
            fail += 1
            print(f"[FAIL] {pdb_id}: {exc}")
    print(f"[PREFETCH] done ok={ok} fail={fail}")


def main():
    parser = argparse.ArgumentParser(description="Prefetch PDBs for a dataset into the local cache.")
    parser.add_argument("--dataset", help="CSV path to use instead of default")
    parser.add_argument("--limit", type=int, help="Optional row limit")
    parser.add_argument(
        "--ppb",
        action="store_true",
        help="Use PPB_Affinity_TCR_pMHC.csv as default dataset",
    )
    args = parser.parse_args()

    default_path = PPB_DATA_CSV if args.ppb else DATA_CSV
    prefetch(args.dataset or default_path, limit=args.limit)


if __name__ == "__main__":
    main()
