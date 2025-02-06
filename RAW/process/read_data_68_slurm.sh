#!/bin/bash
# SBATCH --job-name=read_annot        # Job name
#SBATCH --output=log_slurm/result_read_annot.txt      # Output file
#SBATCH --error=log_slurm/error_read_annot.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=0                 # Number of GPUs per node
python ./read_necrosis_ratio_from_annotations.py