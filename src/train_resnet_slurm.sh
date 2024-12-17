#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=log_slurm/result_resnet_baseline.txt      # Output file
#SBATCH --error=log_slurm/error_resnet_baseline.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=20                              # Number of CPU cores per task                   │····································································································································
conda activate nmduong2
sh ./script/train/baseline/train_vit_baseline.sh