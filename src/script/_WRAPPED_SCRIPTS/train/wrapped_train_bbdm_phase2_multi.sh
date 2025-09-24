#!/bin/bash
#SBATCH --job-name=train_bbdm_multi        # Job name
#SBATCH --output=log_slurm/result/train_bbdm_multi.txt      # Output file
#SBATCH --error=log_slurm/error/train_bbdm_multi.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodes=1               # Số node yêu cầu
#SBATCH --cpus-per-task=8       # Số CPU cho mỗi task
conda init
conda activate nmduong2
cd /home/user01/aiotlab/nmduong/BoneTumor/BBDM2
bash ./script/discreteBBDM/train_discrete_bbdm_multimag.sh