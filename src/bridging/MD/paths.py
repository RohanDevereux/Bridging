from pathlib import Path
import os

PACKAGE_DIR = Path(__file__).resolve().parents[1]

DATA_CSV = PACKAGE_DIR / "processedData" / "PRODIGY_Data.csv"
PPB_DATA_CSV = PACKAGE_DIR / "processedData" / "PPB_Affinity_TCR_pMHC.csv"

GENERATED_DIR = PACKAGE_DIR / "generatedData"
MD_OUT_DIR = GENERATED_DIR / "MD"
PPB_MD_OUT_DIR = GENERATED_DIR / "MD_PPB_TCR_pMHC"
PDB_CACHE_DIR = GENERATED_DIR / "pdb_cache"

# Backwards-compatible aliases for existing modules.
DATA_DIR = PACKAGE_DIR / "processedData"
OUT_DIR = MD_OUT_DIR


def resolve_dataset(path, fallback):
    if path:
        return Path(path)
    env_path = os.getenv("BRIDGING_DATASET")
    if env_path:
        return Path(env_path)
    return fallback
