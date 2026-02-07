import pandas as pd
import requests

from .paths import PDB_CACHE_DIR, PPB_DATA_CSV
from .prefetch_pdbs import RCSB_PDB_URL


def prefetch_all():
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PPB_DATA_CSV)

    pdb_ids = sorted(set(df["PDB"].astype(str).str.upper()))
    for pdb_id in pdb_ids:
        out = PDB_CACHE_DIR / f"{pdb_id}.pdb"
        if out.exists():
            continue

        url = RCSB_PDB_URL.format(pdb=pdb_id)
        response = requests.get(url, timeout=60, headers={"User-Agent": "bridging-md/0.1"})
        response.raise_for_status()
        out.write_bytes(response.content)
        print(f"downloaded {pdb_id}")


if __name__ == "__main__":
    prefetch_all()
