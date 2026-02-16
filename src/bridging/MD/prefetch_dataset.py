import argparse

import pandas as pd
import requests

from .paths import DATA_CSV, PDB_CACHE_DIR, resolve_dataset
from .prefetch_pdbs import ensure_pdb_cached
from bridging.utils.dataset_rows import row_pdb_id


def _collect_pdb_ids(df):
    invalid_samples = []
    invalid_count = 0
    pdb_ids = []

    for row in df.to_dict("records"):
        pdb_id = row_pdb_id(row)
        if pdb_id is not None:
            pdb_ids.append(pdb_id)
            continue

        candidates = []
        for key in ("PDB", "PDB_ID", "complex_pdb"):
            value = row.get(key)
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                candidates.append(text)
        if candidates:
            invalid_count += 1
            if len(invalid_samples) < 20:
                invalid_samples.append(candidates[0])

    return sorted(set(pdb_ids)), invalid_count, invalid_samples


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
    args = parser.parse_args()

    prefetch(args.dataset or DATA_CSV, limit=args.limit)


if __name__ == "__main__":
    main()
