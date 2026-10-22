#!/bin/bash
#SBATCH --job-name=polymerfts-validation
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/%u/polymerfts-validation-%j.log
set -eo pipefail
module purge
module load cuda/12.4.1
source /home/yongdd/anaconda3/etc/profile.d/conda.sh
conda activate polymerfts
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build_dir=$(realpath "${1:?Usage: sbatch --output=<scratch>/validation-%j.log scripts/slurm_validate.sh build report-dir}")
repo_dir=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$build_dir/CMakeCache.txt")
report_dir=$(realpath -m "${2:?A scratch report directory is required}")
mkdir -p "$report_dir"
export PYTHONPATH="$build_dir/python"
export OMP_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=1 OMP_STACKSIZE=1G
ulimit -s unlimited
if [[ "${3:-numerical}" == full ]]; then
    nice -n 10 make -C "$build_dir" -j4
fi
conda list --explicit > "$report_dir/environment-explicit.txt"
cp "$build_dir/CMakeCache.txt" "$report_dir/CMakeCache.txt"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}" nvidia-smi > "$report_dir/nvidia-smi.txt"
python - <<'PY'
from polymerfts import PlatformSelector
available = PlatformSelector.avail_platforms()
if 'cuda' not in available or not any(p.startswith('cpu') for p in available):
    raise SystemExit(f'Validation requires visible CUDA and CPU backends: {available}')
print('Validated backends:', available)
PY
cd "$report_dir"
if [[ "${3:-numerical}" == full ]]; then
    ctest --test-dir "$build_dir" --output-on-failure --output-junit "$report_dir/full.xml" -j1
elif [[ "${3:-numerical}" == failed ]]; then
    ctest --test-dir "$build_dir" --rerun-failed --output-on-failure --output-junit "$report_dir/rerun.xml" -j1
    ctest --test-dir "$build_dir" -R '^(TestParams|TestFtsReliability|TestLftsRun|TestLftsDetails|TestScftResult|TestSamplingStatistics|TestPropagatorAnalytic)$' \
        --output-on-failure --output-junit "$report_dir/focused.xml" -j1
else
    ctest --test-dir "$build_dir" -L numerical --output-on-failure --output-junit "$report_dir/numerical.xml" -j1
fi
python "$repo_dir/examples/scft/Gyroid.py" > "$report_dir/gyroid.log" 2>&1
python - <<'PY'
from scipy.io import loadmat
import json
data = loadmat('fields.mat', squeeze_me=True)
metadata = json.loads(str(data['simulation_metadata_json']))
if metadata['backend'] != 'cuda' or metadata['execution_status'] != 'converged':
    raise SystemExit(f'Gyroid validation failed: {metadata}')
print('Continuous-chain Gyroid converged on CUDA')
PY
