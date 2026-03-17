#!/bin/bash -l

#SBATCH --job-name=graphvae_merge_prep
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=shared
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

DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"

BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-}"
SHARD_ROOT="${SHARD_ROOT:-src/bridging/generatedData/graphvae/${DATASET_STEM}_parallel_prepare}"
MERGED_PREPARED_DIR="${MERGED_PREPARED_DIR:-$SHARD_ROOT/prepared}"

ARGS=(
  --dataset "$DATASET"
  --shard-root "$SHARD_ROOT"
  --out-dir "$MERGED_PREPARED_DIR"
  --train-fraction "${TRAIN_FRACTION:-0.70}"
  --val-fraction "${VAL_FRACTION:-0.15}"
  --split-seed "${SPLIT_SEED:-42}"
)
if [ -n "$BASE_CHECKPOINT_DIR" ]; then
  ARGS+=(--base-checkpoint-dir "$BASE_CHECKPOINT_DIR")
fi

"$PYTHON" -m bridging.graphvae.tools.merge_prepared_shards "${ARGS[@]}"

echo "[DONE] merged_prepared_dir=$MERGED_PREPARED_DIR"
