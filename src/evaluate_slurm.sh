#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=log_slurm/result_evaluate_uni_ft68_discrete_ddim_ce.txt      # Output file
#SBATCH --error=log_slurm/error_evaluate_uni_ft68_discrete_ddim_ce.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=0                 # Number of GPUs per node              │····································································································································
conda activate nmduong2
python ./evaluate.py