#!/bin/bash -l

#SBATCH --job-name=bridging_filter
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
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

# If you use a venv in the repo, uncomment:
# source .venv/bin/activate

PYTHON="$(command -v python3 || command -v python)"
SIZE_CSV="src/bridging/processedData/PPB_Affinity_partner_sizes.csv"
OUT_BASE="src/bridging/processedData/PPB_Affinity_OTHER_large_ca64_target100"

if [ ! -f "$SIZE_CSV" ]; then
  echo "[INFO] size cache not found at $SIZE_CSV; generating..."
  "$PYTHON" -m bridging.dataProcessing.prefetch_partner_sizes \
    --in-csv src/bridging/rawData/PPB-Affinity.csv \
    --out-csv "$SIZE_CSV" \
    --pdb-root src/bridging/generatedData/pdb_cache
fi

"$PYTHON" -m bridging.dataProcessing.filter_ppb_dataset \
  --in-csv src/bridging/rawData/PPB-Affinity.csv \
  --out-csv "${OUT_BASE}_train80_test20.csv" \
  --report-csv "${OUT_BASE}_report.csv" \
  --train-csv "${OUT_BASE}_train.csv" \
  --test-csv "${OUT_BASE}_test.csv" \
  --pdb-root src/bridging/generatedData/pdb_cache \
  --sizes-cache-csv "$SIZE_CSV" \
  --size-source auto \
  --keep-subgroups OTHER \
  --n-interface 64 \
  --max-dN-nm 2.0 \
  --min-partner-len 200 \
  --no-rcsb \
  --target-total-size 100 \
  --train-size 0.8 \
  --test-size 0.2 \
  --split-seed 42 \
  --stratify-by Subgroup
