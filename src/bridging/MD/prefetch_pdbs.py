from pathlib import Path

import pandas as pd
import requests

from .paths import DATA_CSV, PDB_CACHE_DIR

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb}.pdb"


def ensure_pdb_cached(
    pdb_id: str,
    cache_dir: Path = PDB_CACHE_DIR,
    timeout_s: int = 60,
    user_agent: str = "bridging-md/0.1",
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = str(pdb_id).strip().upper()
    out = cache_dir / f"{pdb_id}.pdb"
    if out.exists():
        return out, False

    url = RCSB_PDB_URL.format(pdb=pdb_id)
    response = requests.get(url, timeout=timeout_s, headers={"User-Agent": user_agent})
    response.raise_for_status()
    out.write_bytes(response.content)
    return out, True


def prefetch_all():
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV)

    pdb_ids = sorted(set(df["PDB"].astype(str).str.upper()))
    for pdb_id in pdb_ids:
        _, downloaded = ensure_pdb_cached(pdb_id)
        if downloaded:
            print(f"downloaded {pdb_id}")


if __name__ == "__main__":
    prefetch_all()
