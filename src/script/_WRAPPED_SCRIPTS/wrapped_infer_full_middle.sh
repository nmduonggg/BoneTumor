#!/bin/bash
#SBATCH --job-name=infer_fddm_full_middle        # Job name
#SBATCH --output=log_slurm/result/infer_fddm_full_middle.txt      # Output file
#SBATCH --error=log_slurm/error/infer_fddm_full_middle.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodes=1               # Số node yêu cầu
#SBATCH --cpus-per-task=8       # Số CPU cho mỗi task
conda init
conda activate nmduong2
cd /home/user01/aiotlab/nmduong/BoneTumor/src
bash ./script/PILOT/infer_full_discrete_midstep.sh