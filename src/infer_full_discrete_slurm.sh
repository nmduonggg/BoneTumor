#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=log_slurm/result_nmduong_infer_full_discrete.txt      # Output file
#SBATCH --error=log_slurm/error_nmduong_infer_full_discrete.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=40                              # Number of CPU cores per task                   │····································································································································
conda activate nmduong2
sh ./script/infer_full_discrete.sh
