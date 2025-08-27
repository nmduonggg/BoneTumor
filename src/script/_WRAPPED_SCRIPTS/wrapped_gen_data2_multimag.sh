#!/bin/bash
#SBATCH --job-name=datagen2        # Job name
#SBATCH --output=log_slurm/result/datagen2.txt      # Output file
#SBATCH --error=log_slurm/error/datagen2.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodes=1               # Số node yêu cầu
#SBATCH --cpus-per-task=8       # Số CPU cho mỗi task
conda init
conda activate nmduong2
cd /home/user01/aiotlab/nmduong/BoneTumor/src
bash ./script/PILOT/gen_data/generate_data_2nd_phase_multimag.sh