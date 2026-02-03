#!/bin/bash

#SBATCH --job-name=langevin-fts
#SBATCH --partition=a10 #a100       #normal
#SBATCH --nodes=1
###SBATCH --exclude=syn01
###SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
###SBATCH --gres-flags=enforce-binding
#SBATCH --time=96:00:00
#SBATCH --output %j.out
#SBATCH --error  %j.out
echo "Slurm Job on `hostname`"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo `date`
export OMP_MAX_ACTIVE_LEVELS=0
export OMP_NUM_THREADS=4
# Load modules and run your programs here
python -u Lamella.py $SLURM_JOB_ID
