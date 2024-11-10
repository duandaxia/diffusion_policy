#!/bin/bash
#SBATCH -A berzelius-2024-324
#SBATCH --gpus 1
#SBATCH -t 1-15:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="felicity.duan@gmail.com"

module load Anaconda
module load gcc
conda activate robodiff

cd /proj/daxia/diffusion/diffusion_policy

python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
