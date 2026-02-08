import argparse
import re

import pandas as pd

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
    if "PDB" in df.columns:
        series = df["PDB"]
    elif "PDB_ID" in df.columns:
        series = df["PDB_ID"]
    elif "complex_pdb" in df.columns:
        series = df["complex_pdb"].apply(_pdb_from_complex)
    else:
        raise ValueError("No PDB column found (expected PDB, PDB_ID, or complex_pdb).")

    pdb_ids = (
        pd.Series(series)
        .astype(str)
        .str.strip()
        .str[:4]
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )
    return sorted([p for p in pdb_ids if len(p) == 4])


def prefetch(dataset_path, limit=None):
    dataset_path = resolve_dataset(dataset_path, DATA_CSV)
    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)

    pdb_ids = _collect_pdb_ids(df)
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[PREFETCH] dataset={dataset_path} count={len(pdb_ids)}")
    for pdb_id in pdb_ids:
        _, downloaded = ensure_pdb_cached(pdb_id)
        if downloaded:
            print(f"[OK] {pdb_id}")


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
