from pathlib import Path

from bridging.MD.paths import GENERATED_DIR

MMGBSA_OUT_DIR = GENERATED_DIR / "MMGBSA"


def default_results_path(dataset_path: Path) -> Path:
    return MMGBSA_OUT_DIR / f"{dataset_path.stem}_mmgbsa_estimates.csv"

