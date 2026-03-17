#!/bin/bash -l

#SBATCH --job-name=graphvae_resample
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ucd.ie

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -f "$SLURM_SUBMIT_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
fi

PYTHON="${PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON" ] && [ -x "$HOME/Bridging/.venv/bin/python" ]; then
  PYTHON="$HOME/Bridging/.venv/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || command -v python)"
fi
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1

echo "[ENV] host=$(hostname)"
echo "[ENV] python=$PYTHON"
"$PYTHON" -V
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

if [ -z "${RECORDS:-}" ]; then
  echo "[FAIL] RECORDS is empty."
  exit 1
fi
if [ -z "${OUT_DIR:-}" ]; then
  echo "[FAIL] OUT_DIR is empty."
  exit 1
fi

MODEL_FAMILY="${MODEL_FAMILY:-vae_ridge}"
MODE="${MODE:-S}"
SUPERVISION_MODE="${SUPERVISION_MODE:-unsupervised}"
TARGET_POLICY="${TARGET_POLICY:-shared_static}"
LATENT_DIM="${LATENT_DIM:-16}"
TAG="${TAG:-untagged}"

ARGS=(
  --records "$RECORDS"
  --out-dir "$OUT_DIR"
  --model-family "$MODEL_FAMILY"
  --mode "$MODE"
  --repeats "${REPEATS:-2}"
  --folds "${FOLDS:-5}"
  --val-fraction-of-trainval "${VAL_FRACTION_OF_TRAINVAL:-0.1875}"
  --seed "${SEED:-2026}"
  --device "${DEVICE:-cuda}"
  --hidden-dim "${HIDDEN_DIM:-128}"
  --num-layers "${NUM_LAYERS:-3}"
  --lr "${LR:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-1e-5}"
  --batch-size "${BATCH_SIZE:-16}"
  --max-epochs "${MAX_EPOCHS:-120}"
  --patience "${PATIENCE:-20}"
  --num-workers "${NUM_WORKERS:-4}"
  --checkpoint-every "${TRAIN_CHECKPOINT_EVERY:-1}"
)

if [ -n "${DATASET:-}" ]; then
  ARGS+=(--dataset "$DATASET")
fi
if [ -n "${MMGBSA_CSV:-}" ]; then
  ARGS+=(--mmgbsa "$MMGBSA_CSV")
fi

if [ "$MODEL_FAMILY" = "vae_ridge" ]; then
  ARGS+=(
    --supervision-mode "$SUPERVISION_MODE"
    --affinity-weight "${AFFINITY_WEIGHT:-1.0}"
    --target-policy "$TARGET_POLICY"
    --latent-dim "$LATENT_DIM"
    --mask-ratio "${MASK_RATIO:-0.30}"
    --beta-start "${BETA_START:-0.0}"
    --beta-end "${BETA_END:-1.0}"
    --beta-anneal-fraction "${BETA_ANNEAL_FRACTION:-0.30}"
    --corr-weight "${CORR_WEIGHT:-0.01}"
    --alpha-grid "${ALPHA_GRID:-1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100}"
    --bootstrap "${BOOTSTRAP:-0}"
    --ridge-cv-folds "${RIDGE_CV_FOLDS:-0}"
    --ridge-cv-repeats "${RIDGE_CV_REPEATS:-0}"
    --ridge-cv-inner-folds "${RIDGE_CV_INNER_FOLDS:-5}"
  )
fi

echo "[RUN] tag=$TAG model_family=$MODEL_FAMILY mode=$MODE supervision=$SUPERVISION_MODE target=$TARGET_POLICY latent_dim=$LATENT_DIM"
"$PYTHON" -m bridging.graphvae.resample_config "${ARGS[@]}"
echo "[DONE] tag=$TAG summary=$OUT_DIR/resample_summary.json"
