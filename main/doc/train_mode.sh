#!/bin/bash
#SBATCH -A surendra240700
#SBATCH --reservation ndq
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --output=lstm_l2.txt

module load cuda/10.0
module load cudnn/7.6-cuda-10.0
module add cuda/10.0
module add cudnn/7.6-cuda-10.0
python3 train.py train --dataset=../../train
cp -r /scratch/surendra/logs/lstm ~/lstm_logs/

