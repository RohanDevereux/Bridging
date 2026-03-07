import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]

DATA_CSV = PACKAGE_DIR / "processedData" / "PRODIGY_Data.csv"

GENERATED_DIR = PACKAGE_DIR / "generatedData"
MD_OUT_DIR = GENERATED_DIR / "MD"


def _resolve_scratch_root() -> Path:
    raw = os.getenv("BRIDGING_SCRATCH_ROOT") or os.getenv("SCRATCH_ROOT")
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
    # Default to Sonic-style scratch even if it does not exist yet.
    return (Path.home() / "scratch").expanduser()


SCRATCH_ROOT = _resolve_scratch_root()
PDB_CACHE_DIR = SCRATCH_ROOT / "pdb_cache"

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
