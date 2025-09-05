#!/bin/bash
#SBATCH --job-name=68_train_uni_multimag_ISBI        # Job name
#SBATCH --output=log_slurm/result/train_uni_multimag_ISBI_68.txt      # Output file
#SBATCH --error=log_slurm/error/train_uni_multimag_ISBI_68.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodes=1               # Số node yêu cầu
#SBATCH --cpus-per-task=8       # Số CPU cho mỗi task
conda init
conda activate nmduong2
cd /home/user01/aiotlab/nmduong/BoneTumor/src
bash ./script/PILOT/train/train_uni_lora_cls_heavy_multimag_ISBI_68.sh