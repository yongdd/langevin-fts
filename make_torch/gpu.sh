#$ -V
#$ -q gpu_volta.q
#$ -pe mpi_4 4
#$ -N deep_lfts.sh
#$ -S /bin/bash
#$ -cwd
#$ -l gpu=2
ENV_FILE=$PWD/env_${JOB_ID}
SGE_GPU=$(grep SGE_GPU $ENV_FILE | sed -n "s/SGE_GPU=\(.*\)/\1/p")
export CUDA_VISIBLE_DEVICES=$SGE_GPU
mpirun -np 2 python -u run.py
