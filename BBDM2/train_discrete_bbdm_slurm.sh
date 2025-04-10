#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=log_slurm/result_infer_smooth_vit.txt      # Output file
#SBATCH --error=log_slurm/error_infer_smooth_vit.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=40                              # Number of CPU cores per task                   │····································································································································
conda activate nmduong2
sh ./script/discreteBBDM/train_discrete_bbdm.sh
