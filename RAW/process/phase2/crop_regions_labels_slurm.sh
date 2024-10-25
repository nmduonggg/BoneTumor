#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=log_slurm/result_nmduong_1_crop_data.txt      # Output file
#SBATCH --error=log_slurm/error_nmduong_1_crop_data.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=0                 # Number of GPUs per node
sh ./crop_regions_following_labels.sh