#!/bin/bash -l

#SBATCH --job-name=bridging_md
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=bridging_md_%j.out
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

# Prefetch PDBs (optional but recommended for stability)
# python -m bridging.MD.prefetch_dataset --dataset src/bridging/processedData/antibody_antigen_wt_temp_train50_test20.csv

PYTHON="$(command -v python3 || command -v python)"
"$PYTHON" -m bridging.MD.run_dataset \
  --dataset src/bridging/processedData/antibody_antigen_wt_temp_train50_test20.csv
