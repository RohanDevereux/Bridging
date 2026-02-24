#!/bin/bash -l

#SBATCH --job-name=bridging_ppb_cpu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

echo "[ENV] host=$(hostname)"
echo "[ENV] python=$PYTHON"
"$PYTHON" -V
echo "[ENV] PYTHONPATH=$PYTHONPATH"

DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
MD_ROOT="${MD_ROOT:-$SCRATCH_ROOT/MD_datasets/$DATASET_STEM}"

PPB_PREP_ROOT="${PPB_PREP_ROOT:-$SCRATCH_ROOT/ppb_inputs/$DATASET_STEM}"
PPB_TRAIN_OUT="${PPB_TRAIN_OUT:-src/bridging/generatedData/ppb_results/$DATASET_STEM}"

FRAMES_PER_COMPLEX="${FRAMES_PER_COMPLEX:-120}"
DROP_FIRST_FRACTION="${DROP_FIRST_FRACTION:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_CVFOLDS="${NUM_CVFOLDS:-5}"
SEED="${SEED:-2024}"
MAX_ITERS="${MAX_ITERS:-100000}"
VAL_FREQ="${VAL_FREQ:-1000}"
TRAIN_LOG_FREQ="${TRAIN_LOG_FREQ:-10}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.8}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-2}"
SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-1e-6}"
TRAJ_FRAMES_TRAIN="${TRAJ_FRAMES_TRAIN:-24}"
TRAJ_FRAMES_EVAL="${TRAJ_FRAMES_EVAL:-120}"
TRAJ_BATCH_COMPLEXES="${TRAJ_BATCH_COMPLEXES:-4}"
ONLY_FOLD="${ONLY_FOLD:-}"

mkdir -p "$PPB_PREP_ROOT" "$PPB_TRAIN_OUT"

echo "[PPB] dataset=$DATASET"
echo "[PPB] md_root=$MD_ROOT"
echo "[PPB] prep_out=$PPB_PREP_ROOT"
echo "[PPB] train_out=$PPB_TRAIN_OUT"

"$PYTHON" -m bridging.ppb.prepare_training_data \
  --dataset "$DATASET" \
  --md-root "$MD_ROOT" \
  --out-dir "$PPB_PREP_ROOT" \
  --mode both \
  --frames-per-complex "$FRAMES_PER_COMPLEX" \
  --drop-first-fraction "$DROP_FIRST_FRACTION" \
  --pdb-cache-root "$SCRATCH_ROOT/pdb_cache"

BASELINE_CSV="$PPB_PREP_ROOT/ppb_baseline.csv"
FRAME_AUG_CSV="$PPB_PREP_ROOT/ppb_frame_aug.csv"

if [ ! -f "$BASELINE_CSV" ]; then
  echo "[FAIL] missing baseline csv: $BASELINE_CSV"
  exit 1
fi
if [ ! -f "$FRAME_AUG_CSV" ]; then
  echo "[FAIL] missing frame-aug csv: $FRAME_AUG_CSV"
  exit 1
fi

EXTRA_ARGS=()
if [ -n "$ONLY_FOLD" ]; then
  EXTRA_ARGS+=(--only-fold "$ONLY_FOLD")
fi

"$PYTHON" -m bridging.ppb.train_cv \
  --baseline-csv "$BASELINE_CSV" \
  --frame-aug-csv "$FRAME_AUG_CSV" \
  --out-dir "$PPB_TRAIN_OUT" \
  --device cpu \
  --num-cvfolds "$NUM_CVFOLDS" \
  --max-iters "$MAX_ITERS" \
  --val-freq "$VAL_FREQ" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --train-log-freq "$TRAIN_LOG_FREQ" \
  --scheduler-factor "$SCHEDULER_FACTOR" \
  --scheduler-patience "$SCHEDULER_PATIENCE" \
  --scheduler-min-lr "$SCHEDULER_MIN_LR" \
  --traj-frames-train "$TRAJ_FRAMES_TRAIN" \
  --traj-frames-eval "$TRAJ_FRAMES_EVAL" \
  --traj-batch-complexes "$TRAJ_BATCH_COMPLEXES" \
  "${EXTRA_ARGS[@]}"

echo "[DONE] PPB train complete"
echo "[DONE] prep_report=$PPB_PREP_ROOT/ppb_prepare_report.json"
echo "[DONE] compare_summary=$PPB_TRAIN_OUT/compare_summary.json"
