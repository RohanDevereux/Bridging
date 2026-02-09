#!/bin/bash -l

#SBATCH --job-name=bridging_md
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
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

# If you use a venv in the repo, uncomment:
# source .venv/bin/activate

PYTHON="$(command -v python3 || command -v python)"
DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_diverse100.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"
OUT_ROOT="src/bridging/generatedData/MD_datasets/${DATASET_STEM}"
NUM_SHARDS=6
SHARD_DIR="src/bridging/processedData/shards/${DATASET_STEM}"

# Generate 6 disjoint shards once (task 0), then all array tasks consume one shard each.
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

ROWS=$("$PYTHON" - <<PY
import pandas as pd
df = pd.read_csv("${SHARD_CSV}")
print(len(df))
PY
)
echo "[MD] shard_id=${SHARD_ID} dataset=${SHARD_CSV} rows=${ROWS}"
if [ "$ROWS" -eq 0 ]; then
  echo "[MD] shard is empty; nothing to run."
  exit 0
fi

"$PYTHON" -m bridging.MD.run_dataset \
  --dataset "$SHARD_CSV" \
  --out "$OUT_ROOT"
