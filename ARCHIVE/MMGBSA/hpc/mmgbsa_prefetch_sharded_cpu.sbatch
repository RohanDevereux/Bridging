#!/bin/bash -l

#SBATCH --job-name=mmgbsa_shard
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=shared
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
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

if [ -n "${AMBERTOOLS_BIN:-}" ]; then
  unset AMBERTOOLS_MODULE
fi
if [ -n "${AMBERTOOLS_MODULE:-}" ]; then
  module load "$AMBERTOOLS_MODULE"
fi
if [ -n "${AMBERTOOLS_BIN:-}" ]; then
  export PATH="${AMBERTOOLS_BIN}:$PATH"
fi
echo "[ENV] amberhome=${AMBERHOME:-}"
echo "[ENV] ambertools_bin=${AMBERTOOLS_BIN:-}"
for exe in tleap cpptraj MMPBSA.py; do
  if ! command -v "$exe" >/dev/null 2>&1; then
    echo "[FAIL] missing AmberTools executable: $exe"
    exit 1
  fi
done

DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
MD_ROOT="${MD_ROOT:-$SCRATCH_ROOT/MD_datasets/$DATASET_STEM}"

SHARD_DIR="${MMGBSA_SHARD_DIR:-${SHARD_DIR:-tmp/mmgbsa_shards/$DATASET_STEM}}"
OUT_ROOT="${MMGBSA_OUT_ROOT:-${OUT_ROOT:-src/bridging/generatedData/MMGBSA/${DATASET_STEM}_gb_prodonly_v1}}"
WORK_ROOT="${MMGBSA_WORK_ROOT:-${WORK_ROOT:-$SCRATCH_ROOT/mmgbsa_work/${DATASET_STEM}_gb_prodonly_v1}}"
SHARD_ID="$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"
SHARD_CSV="$SHARD_DIR/shard_${SHARD_ID}.csv"
OUT_CSV="$OUT_ROOT/shard_${SHARD_ID}.csv"
SHARD_WORK_ROOT="$WORK_ROOT/shard_${SHARD_ID}"

if [ ! -f "$SHARD_CSV" ]; then
  echo "[FAIL] missing shard CSV: $SHARD_CSV"
  exit 1
fi

mkdir -p "$OUT_ROOT" "$SHARD_WORK_ROOT"

ROWS=$(( $(wc -l < "$SHARD_CSV") - 1 ))
if [ "$ROWS" -lt 0 ]; then
  ROWS=0
fi
echo "[MMGBSA] shard_id=$SHARD_ID rows=$ROWS csv=$SHARD_CSV out=$OUT_CSV work=$SHARD_WORK_ROOT"
if [ "$ROWS" -eq 0 ]; then
  echo "[MMGBSA] shard is empty; exiting."
  exit 0
fi

ARGS=(
  --dataset "$SHARD_CSV"
  --dataset-label "${DATASET_LABEL:-$DATASET}"
  --out "$OUT_CSV"
  --md-root "$MD_ROOT"
  --work-root "$SHARD_WORK_ROOT"
  --solvation-model "${SOLVATION_MODEL:-gb}"
  --igb "${IGB:-5}"
  --saltcon "${SALTCON:-0.150}"
  --istrng "${ISTRNG:-0.150}"
  --start-frame "${START_FRAME:-21}"
  --interval "${INTERVAL:-5}"
  --source-mode "${SOURCE_MODE:-protein_traj}"
  --no-pdb-prefetch
)
if [ -n "${END_FRAME:-}" ]; then
  ARGS+=(--end-frame "$END_FRAME")
fi
if [ "${REFRESH_MMGBSA:-0}" = "1" ]; then
  ARGS+=(--refresh)
fi
if [ -n "${LIMIT_ROWS:-}" ]; then
  ARGS+=(--limit "$LIMIT_ROWS")
fi

"$PYTHON" -m bridging.MMGBSA.prefetch_dataset "${ARGS[@]}"

echo "[DONE] mmgbsa shard_id=$SHARD_ID out=$OUT_CSV"
