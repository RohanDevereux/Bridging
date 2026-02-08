from pathlib import Path

from bridging.MD.paths import GENERATED_DIR, PACKAGE_DIR
from bridging.featurization.config import FEATURES_BASENAME

DEFAULT_DATASET = PACKAGE_DIR / "processedData" / "antibody_antigen_wt_temp_train50_test20.csv"
DEFAULT_CVAE_CHECKPOINT = Path("models") / "cvae_antibody.pt"
DEFAULT_PRODIGY_DIR = GENERATED_DIR / "PRODIGY"
DEFAULT_OUT_DIR = GENERATED_DIR / "regression"
DEFAULT_FEATURE_FILENAME = FEATURES_BASENAME
