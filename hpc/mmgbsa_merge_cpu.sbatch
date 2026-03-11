#!/bin/bash -l

#SBATCH --job-name=mmgbsa_merge
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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

SHARD_ROOT="${SHARD_ROOT:?SHARD_ROOT is required}"
OUT_CSV="${OUT_CSV:?OUT_CSV is required}"

"$PYTHON" -m bridging.MMGBSA.merge_sharded_results \
  --shard-root "$SHARD_ROOT" \
  --out "$OUT_CSV"

echo "[DONE] merged_mmgbsa=$OUT_CSV"
