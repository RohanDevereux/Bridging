#!/bin/bash -l

#SBATCH --job-name=graphvae_force
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=shared
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

MANIFEST="${MANIFEST:?MANIFEST is required}"
DATASET="${DATASET:?DATASET is required}"
MD_ROOT="${MD_ROOT:?MD_ROOT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
LINE_NO=$(( TASK_ID + 1 ))

ENTRY="$(sed -n "${LINE_NO}p" "$MANIFEST")"
if [ -z "$ENTRY" ]; then
  echo "[FAIL] no manifest entry for task_id=$TASK_ID line_no=$LINE_NO"
  exit 1
fi

IFS=$'\t' read -r LABEL CK_IN CK_OUT REPORT_OUT <<< "$ENTRY"

if [ -z "$CK_IN" ] || [ -z "$CK_OUT" ] || [ -z "$REPORT_OUT" ]; then
  echo "[FAIL] malformed manifest entry: $ENTRY"
  exit 1
fi

IN_COUNT=$(find "$CK_IN" -maxdepth 1 -type f -name 'records_*.pt' | wc -l | awk '{print $1}')
OUT_COUNT=0
if [ -d "$CK_OUT" ]; then
  OUT_COUNT=$(find "$CK_OUT" -maxdepth 1 -type f -name 'records_*.pt' | wc -l | awk '{print $1}')
fi

echo "[FORCE] label=$LABEL in=$CK_IN out=$CK_OUT report=$REPORT_OUT in_count=$IN_COUNT out_count=$OUT_COUNT"

if [ "${REFRESH_FORCE_FEATURES:-0}" != "1" ] && [ "$IN_COUNT" -gt 0 ] && [ "$OUT_COUNT" -eq "$IN_COUNT" ]; then
  echo "[SKIP] force feature part already complete for $LABEL"
  exit 0
fi

"$PYTHON" -m bridging.graphvae.tools.augment_force_features \
  --dataset "$DATASET" \
  --md-root "$MD_ROOT" \
  --pdb-cache-root "${PDB_CACHE_ROOT:-$HOME/scratch/pdb_cache}" \
  --checkpoint-dir-in "$CK_IN" \
  --checkpoint-dir-out "$CK_OUT" \
  --report-out "$REPORT_OUT" \
  --max-frames "${MAX_FRAMES:-120}" \
  --traj-cache-size "${TRAJ_CACHE_SIZE:-1}" \
  --progress-every "${PROGRESS_EVERY:-10}"

echo "[DONE] force feature part label=$LABEL out=$CK_OUT"
