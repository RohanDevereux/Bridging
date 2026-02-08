import pandas as pd

from .paths import PDB_CACHE_DIR, PPB_DATA_CSV
from .prefetch_pdbs import ensure_pdb_cached


def prefetch_all():
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PPB_DATA_CSV)

    pdb_ids = sorted(set(df["PDB"].astype(str).str.upper()))
    for pdb_id in pdb_ids:
        _, downloaded = ensure_pdb_cached(pdb_id)
        if downloaded:
            print(f"downloaded {pdb_id}")


if __name__ == "__main__":
    prefetch_all()
