#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

python3 run.py
