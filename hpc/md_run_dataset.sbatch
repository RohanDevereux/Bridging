#!/bin/bash -l

#SBATCH --job-name=bridging_md
#SBATCH --time=72:00:00
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

# Use repo venv when available (avoids node-specific system Python issues).
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

PYTHON="$(command -v python3 || command -v python)"
DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_TCR_len200_ca64_298_train80_test20.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"
OUT_ROOT="src/bridging/generatedData/MD_datasets/${DATASET_STEM}"
NUM_SHARDS=2
SHARD_DIR="src/bridging/processedData/shards/${DATASET_STEM}"
READY_FILE="${SHARD_DIR}/.ready"

# Generate 2 disjoint shards once (task 0), then all array tasks consume one shard each.
if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
  rm -rf "$SHARD_DIR"
  mkdir -p "$SHARD_DIR"
  "$PYTHON" - <<PY
from pathlib import Path
import csv

dataset = Path("${DATASET}")
shard_dir = Path("${SHARD_DIR}")
ready_file = Path("${READY_FILE}")
num_shards = int("${NUM_SHARDS}")

with dataset.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)
if not fieldnames:
    raise RuntimeError(f"No CSV header found in dataset: {dataset}")

pdb_col = None
for c in ("PDB", "pdb", "PDB_ID", "pdb_id"):
    if c in fieldnames:
        pdb_col = c
        break
if pdb_col is not None:
    deduped = []
    seen = set()
    for row in rows:
        pdb = str(row.get(pdb_col, "")).strip().upper()
        row[pdb_col] = pdb
        if pdb in seen:
            continue
        seen.add(pdb)
        deduped.append(row)
    rows = deduped

shard_rows = [[] for _ in range(num_shards)]
for idx, row in enumerate(rows):
    shard_rows[idx % num_shards].append(row)

shard_pdb_sets = []
for shard_id, out_rows in enumerate(shard_rows):
    out_path = shard_dir / f"shard_{shard_id:02d}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[SHARD] id={shard_id} rows={len(out_rows)} out={out_path}")
    if pdb_col is not None:
        shard_pdb_sets.append(
            set(str(r.get(pdb_col, "")).strip().upper() for r in out_rows)
        )

if pdb_col is not None:
    overlap = set()
    for i in range(len(shard_pdb_sets)):
        for j in range(i + 1, len(shard_pdb_sets)):
            overlap |= shard_pdb_sets[i].intersection(shard_pdb_sets[j])
    if overlap:
        raise RuntimeError(f"Shard overlap detected for PDB IDs: {sorted(overlap)[:10]}")

ready_file.write_text("ok\n", encoding="utf-8")
print(f"[SHARD] ready={ready_file}")
PY
fi

SHARD_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_CSV="${SHARD_DIR}/shard_$(printf '%02d' "$SHARD_ID").csv"

# Wait for shard files to be materialized by task 0.
for _ in $(seq 1 120); do
  if [ -f "$READY_FILE" ] && [ -f "$SHARD_CSV" ]; then
    break
  fi
  sleep 2
done
if [ ! -f "$READY_FILE" ] || [ ! -f "$SHARD_CSV" ]; then
  echo "[FAIL] shard setup incomplete ready=$READY_FILE shard=$SHARD_CSV"
  exit 1
fi

ROWS=$("$PYTHON" - <<PY
import csv
with open("${SHARD_CSV}", "r", encoding="utf-8", newline="") as f:
    r = csv.reader(f)
    next(r, None)
    print(sum(1 for _ in r))
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
