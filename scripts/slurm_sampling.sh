#!/bin/bash
#SBATCH --job-name=polymerfts-sampling
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/%u/polymerfts-sampling-%j.log
# CPU-only jobs intentionally request no GPUs.
set -eo pipefail
module purge
module load cuda/12.4.1
source /home/yongdd/anaconda3/etc/profile.d/conda.sh
conda activate polymerfts
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
repo_dir="${POLYMERFTS_SOURCE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$repo_dir/scripts/validate_fts_sampling.py" && -f "$repo_dir/CMakeCache.txt" ]]; then
    repo_dir=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$repo_dir/CMakeCache.txt")
fi
export OMP_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=1 OMP_STACKSIZE=1G
export CUDA_VISIBLE_DEVICES=""
ulimit -s unlimited
python "$repo_dir/scripts/validate_fts_sampling.py" "$@"
