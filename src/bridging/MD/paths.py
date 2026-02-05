from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]

DATA_CSV = PACKAGE_DIR / "processedData" / "PRODIGY_Data.csv"

GENERATED_DIR = PACKAGE_DIR / "generatedData"
MD_OUT_DIR = GENERATED_DIR / "MD"
PDB_CACHE_DIR = GENERATED_DIR / "pdb_cache"

# Backwards-compatible aliases for existing modules.
DATA_DIR = PACKAGE_DIR / "processedData"
OUT_DIR = MD_OUT_DIR
