#!/bin/bash

#SBATCH --job-name=pre
#SBATCH --time=2-00:00:00
#SBATCH --account=plgdyplomancipw-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu
# #SBATCH --exclude=t0033


gpu_id=$CUDA_VISIBLE_DEVICES
echo $gpu_id

echo `pwd`
source ../../.venv/bin/activate
echo # here args
bash # here script
echo "fin"