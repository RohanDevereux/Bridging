#!/bin/bash -l

#SBATCH --job-name=graphvae_sweep
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
  source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
fi

PYTHON="${PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
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
  echo "[HINT] Set RECORDS to a prepared graph_records.pt path."
  exit 1
fi

OUT_DIR="${OUT_DIR:-src/bridging/generatedData/graphvae/sweeps/default}"
GRAPH_VIEWS="${GRAPH_VIEWS:-full}"
MODES="${MODES:-${MODE:-S,SD}}"
LATENT_DIMS="${LATENT_DIMS:-8,16,32,64}"
SUPERVISION_MODES="${SUPERVISION_MODES:-unsupervised,semi_supervised}"
AFFINITY_WEIGHT="${AFFINITY_WEIGHT:-1.0}"

ARGS=(
  --records "$RECORDS"
  --out-dir "$OUT_DIR"
  --graph-views "$GRAPH_VIEWS"
  --modes "$MODES"
  --latent-dims "$LATENT_DIMS"
  --supervision-modes "$SUPERVISION_MODES"
  --affinity-weight "$AFFINITY_WEIGHT"
  --device "${DEVICE:-cuda}"
  --hidden-dim "${HIDDEN_DIM:-128}"
  --num-layers "${NUM_LAYERS:-3}"
  --mask-ratio "${MASK_RATIO:-0.30}"
  --lr "${LR:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-1e-5}"
  --batch-size "${BATCH_SIZE:-16}"
  --max-epochs "${MAX_EPOCHS:-120}"
  --patience "${PATIENCE:-20}"
  --beta-start "${BETA_START:-0.0}"
  --beta-end "${BETA_END:-1.0}"
  --beta-anneal-fraction "${BETA_ANNEAL_FRACTION:-0.30}"
  --corr-weight "${CORR_WEIGHT:-0.01}"
  --seed "${SEED:-2026}"
  --num-workers "${NUM_WORKERS:-4}"
  --checkpoint-every "${TRAIN_CHECKPOINT_EVERY:-1}"
  --alpha-grid "${ALPHA_GRID:-1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100}"
  --bootstrap "${BOOTSTRAP:-0}"
  --ridge-cv-folds "${RIDGE_CV_FOLDS:-5}"
  --ridge-cv-repeats "${RIDGE_CV_REPEATS:-3}"
  --ridge-cv-inner-folds "${RIDGE_CV_INNER_FOLDS:-5}"
)

if [ -n "${DATASET:-}" ]; then
  ARGS+=(--dataset "$DATASET")
fi
if [ -n "${MMGBSA_CSV:-}" ]; then
  ARGS+=(--mmgbsa "$MMGBSA_CSV")
fi

"$PYTHON" -m bridging.graphvae.sweep "${ARGS[@]}"

echo "[DONE] sweep summary: $OUT_DIR/sweep_summary.json"
