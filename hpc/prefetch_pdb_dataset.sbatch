#!/bin/bash -l

#SBATCH --job-name=bridging_prefetch_pdb
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
DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_diverse100.csv}"

"$PYTHON" -m bridging.MD.prefetch_dataset \
  --dataset "$DATASET"
