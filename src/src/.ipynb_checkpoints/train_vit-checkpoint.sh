#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=log_slurm/result_nmduong.txt      # Output file
#SBATCH --error=log_slurm/error_nmduong.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=20                              # Number of CPU cores per task                   │····································································································································

python train_classifier.py \
    -opt options/ViT_baseline.yml \