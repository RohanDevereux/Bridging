#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --job-name=prodigy_md

cd $SLURM_SUBMIT_DIR
python -m bridging.MD.run
