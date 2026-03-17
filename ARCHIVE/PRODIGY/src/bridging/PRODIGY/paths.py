from pathlib import Path

from bridging.MD.paths import GENERATED_DIR

PRODIGY_OUT_DIR = GENERATED_DIR / "PRODIGY"


def default_results_path(dataset_path: Path) -> Path:
    return PRODIGY_OUT_DIR / f"{dataset_path.stem}_prodigy_estimates.csv"
