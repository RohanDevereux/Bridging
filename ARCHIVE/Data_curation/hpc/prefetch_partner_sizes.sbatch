#!/bin/bash -l

#SBATCH --job-name=bridging_sizes
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
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
BRIDGING_SCRATCH_ROOT="${BRIDGING_SCRATCH_ROOT:-$SCRATCH_ROOT}"
export BRIDGING_SCRATCH_ROOT
PDB_ROOT="${PDB_ROOT:-$BRIDGING_SCRATCH_ROOT/pdb_cache}"
mkdir -p "$PDB_ROOT"
echo "[PATHS] scratch=$BRIDGING_SCRATCH_ROOT pdb_root=$PDB_ROOT"
"$PYTHON" -m bridging.dataProcessing.prefetch_partner_sizes \
  --in-csv src/bridging/rawData/PPB-Affinity.csv \
  --out-csv src/bridging/processedData/PPB_Affinity_partner_sizes.csv \
  --pdb-root "$PDB_ROOT"
