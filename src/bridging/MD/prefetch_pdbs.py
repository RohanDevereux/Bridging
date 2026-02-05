from pathlib import Path

import pandas as pd
import requests

from .paths import DATA_CSV, PDB_CACHE_DIR

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb}.pdb"


def prefetch_all():
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV)

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
