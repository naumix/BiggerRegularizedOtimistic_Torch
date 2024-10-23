#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate bro_torch

module load cuDNN/8.9.2.26-CUDA-12.2.0

python3 train_torch.py --env_name=fish-swim & python3 train_torch.py --env_name=fish-swim & python3 train_torch.py --env_name=fish-swim

wait
