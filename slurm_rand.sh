#!/bin/bash
#
#SBATCH --job-name=tsp-pointer
#SBATCH --output=logs/tsp_%A.out
#SBATCH --error=logs/tsp_%A.err
#
#SBATCH --time=30-00:00:00
#SBATCH --nodelist=komputasi08

source ~/ttp/bin/activate;
srun python train.py --title rand_not_learnable &
srun python train.py --learnable-first-input True --title rand_learnable &
srun tensorboard --logdir=runs &

wait;
