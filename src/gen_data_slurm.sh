#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=log_slurm/result_nmduong_gen.txt      # Output file
#SBATCH --error=log_slurm/error_nmduong_gen.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=20                              # Number of CPU cores per task                   │····································································································································
conda activate nmduong2
sh ./script/gen_data/generate_data_2nd_phase.sh
