from pathlib import Path

from ..MD.paths import MD_OUT_DIR

FEATURES_FILENAME = "contact_distance_fp16.npy"
DEFAULT_FEATURE_GLOB = str(MD_OUT_DIR / "*" / "features" / FEATURES_FILENAME)

LATENT_DIM = 32
BASE_CHANNELS = 32
ENCODER_TYPE = "set_edge"
SET_HIDDEN = 64
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
BETA_MAX = 1.0
BETA_WARMUP_EPOCHS = 10
