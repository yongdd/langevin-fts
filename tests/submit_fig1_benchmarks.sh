#!/bin/bash
#
# Submit Fig. 1 benchmark jobs for all methods and Ns values
# Usage: ./submit_fig1_benchmarks.sh
#
# This script submits SLURM jobs to run benchmark_song2018.py for all
# combinations of numerical methods and Ns values in parallel.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Methods to test
METHODS=("rqm4" "etdrk4" "cn-adi2" "cn-adi4")

# Ns values (contour steps) - including non-integer block sizes (fixed)
NS_VALUES=(40 80 100 160 200 320 400 640 800 1000 2000 4000)

# SLURM partition (adjust for your cluster)
PARTITION="a10"

echo "========================================"
echo "Submitting Fig. 1 Benchmark Jobs"
echo "========================================"
echo "Methods: ${METHODS[*]}"
echo "Ns values: ${NS_VALUES[*]}"
echo "Partition: ${PARTITION}"
echo ""

# Create logs directory
mkdir -p logs

# Submit jobs
for method in "${METHODS[@]}"; do
    for ns in "${NS_VALUES[@]}"; do
        job_name="fig1_${method}_Ns${ns}"
        output_file="logs/${job_name}_%j.out"

        echo "Submitting: $job_name"

        sbatch --job-name="$job_name" \
               --partition="$PARTITION" \
               --nodes=1 \
               --gres=gpu:1 \
               --cpus-per-task=4 \
               --time=02:00:00 \
               --output="$output_file" \
               --error="$output_file" \
               --wrap="export OMP_MAX_ACTIVE_LEVELS=0; export OMP_NUM_THREADS=4; python -u benchmark_song2018.py fig1 $method --Ns $ns"
    done
done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
echo "Results will be saved to: benchmark_fig1_<method>_Ns<N>.json"
