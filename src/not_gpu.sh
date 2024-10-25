#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=log_slurm/result_nmduong_1_send_data.txt      # Output file
#SBATCH --error=log_slurm/error_nmduong_1_send_data.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=0                 # Number of GPUs per node
scp -r ./RAW_DATA aiotlab@222.252.4.92:/mnt/disk4/nmduong/Vin-Uni-Bone-Tumor/RAW_DATA
