#!/bin/bash -l

#SBATCH --job-name=bridging_ml
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=bridging_ml_%j.out
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
"$PYTHON" -m bridging.ml.train \
  --features "src/bridging/generatedData/MD/*/features/contact_distance_fp16.npy" \
  --out models/cvae_antibody.pt
