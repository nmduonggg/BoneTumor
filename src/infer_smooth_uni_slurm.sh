#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=log_slurm/result_infer_smooth_uni_68.txt      # Output file
#SBATCH --error=log_slurm/error_infer_smooth_uni_68.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=20                              # Number of CPU cores per task                   │····································································································································
conda activate nmduong2
sh ./script/infer_baseline/infer_smooth_uni.sh
