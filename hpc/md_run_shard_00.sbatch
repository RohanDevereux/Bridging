#!/bin/bash -l

#SBATCH --job-name=bridging_md_s0
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@ucd.ie

cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

module avail >/dev/null 2>&1
module load cuda 2>/dev/null || true

command -v nvidia-smi || true
nvidia-smi || true

if [ -f "$SLURM_SUBMIT_DIR/.venv/bin/activate" ]; then
  source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
fi

PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || command -v python)"
fi
echo "[ENV] python=$PYTHON"
"$PYTHON" -V

DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"
SHARD_CSV="src/bridging/processedData/shards/${DATASET_STEM}/shard_00.csv"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
BRIDGING_SCRATCH_ROOT="${BRIDGING_SCRATCH_ROOT:-$SCRATCH_ROOT}"
export BRIDGING_SCRATCH_ROOT
OUT_ROOT="${OUT_ROOT:-$SCRATCH_ROOT/MD_datasets/${DATASET_STEM}}"
PDB_CACHE_DIR="${PDB_CACHE_DIR:-$BRIDGING_SCRATCH_ROOT/pdb_cache}"
mkdir -p "$PDB_CACHE_DIR"
mkdir -p "$OUT_ROOT"
echo "[PATHS] scratch=$BRIDGING_SCRATCH_ROOT pdb_cache=$PDB_CACHE_DIR out=$OUT_ROOT"

if [ ! -f "$SHARD_CSV" ]; then
  echo "[FAIL] missing shard CSV: $SHARD_CSV"
  echo "[HINT] run preshard first:"
  echo "  PYTHONPATH=src python -m bridging.dataProcessing.preshard_dataset --dataset $DATASET --num-shards 2"
  exit 1
fi

ROWS=$(( $(wc -l < "$SHARD_CSV") - 1 ))
if [ "$ROWS" -lt 0 ]; then
  ROWS=0
fi
echo "[MD] shard=00 dataset=$SHARD_CSV rows=$ROWS"
if [ "$ROWS" -eq 0 ]; then
  echo "[MD] shard is empty; nothing to run."
  exit 0
fi

"$PYTHON" -m bridging.MD.run_dataset \
  --dataset "$SHARD_CSV" \
  --out "$OUT_ROOT"
