#!/usr/bin/env bash
# Build a versioned wheel of polymerfts and install it into its own virtual
# environment, so that several versions can be kept side by side.
#
#   scripts/install_versioned.sh                 # wheel + venv for the version in pyproject.toml
#   scripts/install_versioned.sh --skip-build    # reuse an existing wheel from the wheel directory
#
# Layout (override with --wheel-dir / --venv-dir):
#   ~/wheels/polymerfts-<version>-<tag>.whl      the build artefact, kept for later reinstalls
#   ~/venvs/polymerfts-<version>/                venv created on top of the active Python
#                                                (--system-site-packages: numpy, scipy, MKL, ... are
#                                                shared with the conda environment; only polymerfts
#                                                lives in the venv)
# Use a version with:
#   source ~/venvs/polymerfts-<version>/bin/activate
#
# Run it with the conda environment active that provides CMake, MKL/FFTW, spglib
# and LAPACKE, and with a CUDA 12.2+ nvcc first on PATH for the GPU backend (the
# conda environment has no nvcc); the build takes the same options as the CMake
# build (see --mkl/--fftw/--cuda; --cuda AUTO builds without the GPU backend when
# no usable toolkit is found, ON stops instead).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="${HOME}/wheels"
VENV_DIR="${HOME}/venvs"
USE_MKL=ON
USE_FFTW=ON
USE_CUDA=ON
JOBS=4
SKIP_BUILD=0
EXTRA_PIP_ARGS=()

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wheel-dir) WHEEL_DIR="$2"; shift 2 ;;
        --venv-dir)  VENV_DIR="$2";  shift 2 ;;
        --mkl)       USE_MKL="$2";   shift 2 ;;   # ON or OFF
        --fftw)      USE_FFTW="$2";  shift 2 ;;
        --cuda)      USE_CUDA="$2";  shift 2 ;;
        -j|--jobs)   JOBS="$2";      shift 2 ;;
        --skip-build) SKIP_BUILD=1;  shift ;;
        --no-build-isolation) EXTRA_PIP_ARGS+=(--no-build-isolation); shift ;;
        -h|--help)   usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

PYTHON="$(command -v python)"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "${ROOT}/pyproject.toml")"
[[ -n "${VERSION}" ]] || { echo "could not read the version from pyproject.toml" >&2; exit 1; }
echo "polymerfts ${VERSION}  (python: ${PYTHON})"

# 1. Wheel -----------------------------------------------------------------
mkdir -p "${WHEEL_DIR}"
if [[ "${SKIP_BUILD}" -eq 0 && "${USE_CUDA}" != "OFF" ]]; then
    # scikit-build-core keeps its build directories (build/<wheel tag>, see
    # pyproject.toml). One configured with another nvcc keeps that compiler
    # cached, so start it afresh when the nvcc on PATH has changed.
    NVCC="$(command -v nvcc || true)"
    for CACHE in "${ROOT}"/build/cp*-*/CMakeCache.txt; do
        [[ -f "${CACHE}" ]] || continue
        CACHED="$(sed -n 's/^CMAKE_CUDA_COMPILER:[A-Z]*=//p' "${CACHE}")"
        if [[ -n "${CACHED}" && -n "${NVCC}" && "$(readlink -f "${CACHED}")" != "$(readlink -f "${NVCC}")" ]]; then
            echo "== ${CACHE%/*} was configured with ${CACHED}; reconfiguring with ${NVCC}"
            rm -f "${CACHE}"
            rm -rf "${CACHE%/*}/CMakeFiles"
        fi
    done
fi
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo "== building wheel into ${WHEEL_DIR} (MKL=${USE_MKL} FFTW=${USE_FFTW} CUDA=${USE_CUDA}, ${JOBS} jobs)"
    CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    CMAKE_PREFIX_PATH="${CONDA_PREFIX:-}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}" \
    "${PYTHON}" -m pip wheel "${ROOT}" --no-deps -w "${WHEEL_DIR}" "${EXTRA_PIP_ARGS[@]}" \
        -C "cmake.define.POLYMERFTS_USE_MKL=${USE_MKL}" \
        -C "cmake.define.POLYMERFTS_USE_FFTW=${USE_FFTW}" \
        -C "cmake.define.POLYMERFTS_USE_CUDA=${USE_CUDA}"
fi
WHEEL="$(ls -t "${WHEEL_DIR}"/polymerfts-"${VERSION}"-*.whl 2>/dev/null | head -1 || true)"
[[ -n "${WHEEL}" ]] || { echo "no wheel for ${VERSION} in ${WHEEL_DIR}" >&2; exit 1; }
echo "== wheel: ${WHEEL}"

# 2. Virtual environment on top of the active Python --------------------------
VENV="${VENV_DIR}/polymerfts-${VERSION}"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "== creating ${VENV}"
    "${PYTHON}" -m venv --system-site-packages "${VENV}"
fi

# 3. Install the wheel into the venv ------------------------------------------
echo "== installing into ${VENV}"
"${VENV}/bin/python" -m pip install --no-deps --force-reinstall --quiet "${WHEEL}"
"${VENV}/bin/python" - <<'PY'
import polymerfts
print(f"== installed polymerfts {polymerfts.__version__} at {polymerfts.__file__}")
print(f"   platforms: {polymerfts.PlatformSelector.avail_platforms()}")
PY
echo
echo "Use it with:  source ${VENV}/bin/activate"
