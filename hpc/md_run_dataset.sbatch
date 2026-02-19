#!/bin/bash -l

#SBATCH --job-name=bridging_md
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@ucd.ie

cd $SLURM_SUBMIT_DIR

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

# If your cluster uses modules for CUDA/NVIDIA tools
module avail >/dev/null 2>&1
module load cuda 2>/dev/null || true

command -v nvidia-smi || true
nvidia-smi || true

# Prefer repo venv for consistent interpreter across nodes.
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
PARENT_DIR="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$PARENT_DIR/scratch}"
BRIDGING_SCRATCH_ROOT="${BRIDGING_SCRATCH_ROOT:-$SCRATCH_ROOT}"
export BRIDGING_SCRATCH_ROOT
OUT_ROOT="${OUT_ROOT:-$SCRATCH_ROOT/MD_datasets/${DATASET_STEM}}"
PDB_CACHE_DIR="${PDB_CACHE_DIR:-$BRIDGING_SCRATCH_ROOT/pdb_cache}"
mkdir -p "$PDB_CACHE_DIR"
NUM_SHARDS=2
SHARD_DIR="src/bridging/processedData/shards/${DATASET_STEM}"
mkdir -p "$OUT_ROOT"
echo "[PATHS] scratch=$BRIDGING_SCRATCH_ROOT pdb_cache=$PDB_CACHE_DIR out=$OUT_ROOT"

# Generate 2 disjoint shards once (task 0), then all array tasks consume one shard each.
if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
  mkdir -p "$SHARD_DIR"
  "$PYTHON" - <<PY
from pathlib import Path
import pandas as pd

dataset = Path("${DATASET}")
shard_dir = Path("${SHARD_DIR}")
num_shards = int("${NUM_SHARDS}")

df = pd.read_csv(dataset)
pdb_col = None
for c in ("PDB", "pdb", "PDB_ID", "pdb_id"):
    if c in df.columns:
        pdb_col = c
        break
if pdb_col is not None:
    df[pdb_col] = df[pdb_col].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=[pdb_col], keep="first").reset_index(drop=True)

shard_indices = [[] for _ in range(num_shards)]
for idx in range(len(df)):
    shard_indices[idx % num_shards].append(idx)

for shard_id, idxs in enumerate(shard_indices):
    out = df.iloc[idxs].copy()
    out_path = shard_dir / f"shard_{shard_id:02d}.csv"
    out.to_csv(out_path, index=False)
    print(f"[SHARD] id={shard_id} rows={len(out)} out={out_path}")
PY
fi

SHARD_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_CSV="${SHARD_DIR}/shard_$(printf '%02d' "$SHARD_ID").csv"

# Wait for shard files to be materialized by task 0.
for _ in $(seq 1 120); do
  if [ -f "$SHARD_CSV" ]; then
    break
  fi
  sleep 2
done
if [ ! -f "$SHARD_CSV" ]; then
  echo "[FAIL] shard not found: $SHARD_CSV"
  exit 1
fi

# Count data rows without importing pandas again.
ROWS=$(( $(wc -l < "$SHARD_CSV") - 1 ))
if [ "$ROWS" -lt 0 ]; then
  ROWS=0
fi
echo "[MD] shard_id=${SHARD_ID} dataset=${SHARD_CSV} rows=${ROWS}"
if [ "$ROWS" -eq 0 ]; then
  echo "[MD] shard is empty; nothing to run."
  exit 0
fi

"$PYTHON" -m bridging.MD.run_dataset \
  --dataset "$SHARD_CSV" \
  --out "$OUT_ROOT"
