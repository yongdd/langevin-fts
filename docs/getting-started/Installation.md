# Installation

You need Conda and a C++20 compiler. Start by cloning the repository.

## 1. Create the Environment

```bash
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts
conda env create -f environment.yml
conda activate polymerfts
```

The environment uses Anaconda's official channel and Python 3.14; spglib is
installed from its upstream PyPI wheel.

## 2. Build a Wheel with MKL or FFTW

Run **one** of the following blocks. Neither FFT library is included in the base
environment.

For GPU support, make CUDA 12.2+ `nvcc` available on PATH **before this step**.
CUDA is then detected automatically; see [GPU setup](#gpu-setup) for details.

**MKL**

```bash
conda install --override-channels -c defaults --freeze-installed \
    mkl=2025.0.0 mkl-include=2025.0.0
CMAKE_PREFIX_PATH="$CONDA_PREFIX" python -m pip wheel . --no-deps --wheel-dir dist \
    -Ccmake.define.POLYMERFTS_USE_MKL=ON -Ccmake.define.POLYMERFTS_USE_FFTW=OFF
```

**FFTW**

```bash
conda install --override-channels -c defaults --freeze-installed fftw
CMAKE_PREFIX_PATH="$CONDA_PREFIX" python -m pip wheel . --no-deps --wheel-dir dist \
    -Ccmake.define.POLYMERFTS_USE_MKL=OFF -Ccmake.define.POLYMERFTS_USE_FFTW=ON
```

FFTW-enabled binary distributions must comply with its
[GPL terms](#gpl-license-warning-fftw).

## 3. Install and Verify

```bash
python -m pip install --no-deps --force-reinstall dist/polymerfts-0.10.0-*.whl
python -c "from polymerfts import PlatformSelector; print(PlatformSelector.avail_platforms())"
```

You can now run the [Quick Start](QuickStart.md) examples.
Rebuild and reinstall the wheel to update the package. Pip tracks its installed
files, including removal of modules dropped by a newer version.

## Optional Settings

<details>
<summary>GPU setup</summary>

### GPU Setup

Install a CUDA 12.2+ toolkit and compatible NVIDIA driver. The Conda environment
does not include `nvcc`. For example, before building:

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
```

The default `POLYMERFTS_USE_CUDA=AUTO` builds CUDA when the toolchain is usable.
Use `ON` to require CUDA or `OFF` for a CPU-only build. A CUDA-only configuration
needs neither MKL nor FFTW:

```bash
CMAKE_PREFIX_PATH="$CONDA_PREFIX" python -m pip wheel . --no-deps --wheel-dir dist \
    -Ccmake.define.POLYMERFTS_USE_MKL=OFF -Ccmake.define.POLYMERFTS_USE_FFTW=OFF \
    -Ccmake.define.POLYMERFTS_USE_CUDA=ON
```

To see the available platforms after installation:

```bash
python -c "from polymerfts import PlatformSelector; print(PlatformSelector.avail_platforms())"
```

`cuda` requires both a compiled CUDA backend and a visible GPU.

</details>

<details>
<summary>Dependencies and CMake options</summary>

### Dependencies

`environment.yml` supplies the shared dependencies. Install the selected CPU FFT
library separately, and provide the compiler and any CUDA toolkit outside Conda.

| Dependency | Requirement |
|------------|-------------|
| Python | 3.11+; the supplied environment uses 3.14 |
| C++ compiler | C++20 with `std::source_location`, such as GCC 11+; CUDA also requires a compatible host compiler |
| CMake | 3.18+ for CPU-only builds; 3.25.2+ for CUDA |
| spglib | 2.x; included in the environment |
| LAPACKE | Included in the environment; manual installs can use `liblapacke-dev` or `LAPACKE_ROOT` |
| MKL / FFTW | Install either CPU FFT library using step 2 |
| CUDA toolkit and driver | Optional; CUDA 12.2+ and a compatible driver |

### CMake Options

Pass wheel build options as `-Ccmake.define.OPTION=VALUE` to `pip wheel`.

| Option | Default | Purpose |
|--------|---------|---------|
| `POLYMERFTS_USE_MKL` | OFF | MKL CPU backend |
| `POLYMERFTS_USE_FFTW` | OFF | FFTW CPU backend (GPL) |
| `POLYMERFTS_USE_CUDA` | AUTO | Detect CUDA; `ON` requires it, `OFF` disables it |
| `POLYMERFTS_CUDA_ARCHITECTURES` | empty | Semicolon-separated GPU capabilities; empty selects defaults for the toolkit |
| `POLYMERFTS_BUILD_TESTS` | ON | Build tests |
| `POLYMERFTS_INSTALL_PYTHON` | ON | Install the Python package |
| `POLYMERFTS_WERROR` | OFF | Treat host C++ warnings as errors |
| `POLYMERFTS_INSTALL_HEADERS` | OFF | Install the experimental downstream C++ interface |

CMake defaults to a Release build. For the full configuration, see
[CMakeLists.txt](../../CMakeLists.txt).

</details>

<details>
<summary>Keeping several versions side by side</summary>

### Keeping Several Versions Side by Side

Keep the wheel built in step 2 and install each version into its own virtual
environment. Replace the version below with that checkout's version:

```bash
python -m venv --system-site-packages /path/to/venvs/polymerfts-0.10.0
/path/to/venvs/polymerfts-0.10.0/bin/python -m pip install --no-deps --force-reinstall \
    dist/polymerfts-0.10.0-*.whl
source /path/to/venvs/polymerfts-0.10.0/bin/activate
```

Keep the Conda environment available: `--system-site-packages` shares its
dependencies. Releases before 0.10.0 use their own CMake installation instructions.

</details>

## Testing

| Check | Command |
|-------|---------|
| Installed wheel | `python tests/TestPropagatorAnalytic.py` |
| Native regression suite | `ctest --test-dir build-tests --output-on-failure` |

The native build below stages its package in `build-tests/python` for CTest.
Available tests depend on the enabled backends. CUDA tests need a visible GPU.
See [Numerical Validation](../internals/NumericalValidation.md) for coverage.

<details>
<summary>Development: test MKL and FFTW together</summary>

CPU CI enables both backends in one job, including their agreement checks.
To use the same configuration locally:

```bash
conda install --override-channels -c defaults --freeze-installed \
    mkl=2025.0.0 mkl-include=2025.0.0 fftw
cmake -S . -B build-tests -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DPOLYMERFTS_USE_MKL=ON -DPOLYMERFTS_USE_FFTW=ON \
    -DPOLYMERFTS_USE_CUDA=OFF -DPOLYMERFTS_WERROR=ON
cmake --build build-tests --parallel 4
ctest --test-dir build-tests --output-on-failure
```

</details>

## Troubleshooting

### CUDA Not Detected

Put CUDA 12.2+ `nvcc` on PATH and rebuild the wheel. The configure warning beginning
`Building without the GPU backend:` explains why CUDA was disabled. Use
`-Ccmake.define.POLYMERFTS_USE_CUDA=ON` to make this a configuration error.

### CUDA Toolkit Too Old

Select a CUDA 12.2+ toolkit and rebuild the wheel; CUDA builds also need CMake 3.25.2+.
The build links the cuFFT library belonging to the selected toolkit.

### nvcc Rejects the Host Compiler

Choose a compatible compiler with `-Ccmake.define.CMAKE_CUDA_HOST_COMPILER=/path/to/g++`.
For example, nvcc 12.2 rejects GCC 13, while nvcc 12.4 accepts it. Compiler errors
are recorded in the wheel build directory's `CMakeFiles/CMakeConfigureLog.yaml`.

### Unsupported GPU Architecture

Add `-Ccmake.define.POLYMERFTS_CUDA_ARCHITECTURES="80;86"` to the wheel command,
using capabilities supported by your toolkit and GPU. When changing toolkits,
use a fresh build directory with `-Cbuild-dir=build/wheel-new-toolkit`.

### Segmentation Fault

If the failure is caused by insufficient stack space, increase the limits:

```bash
ulimit -s unlimited
export OMP_STACKSIZE=1G
```

## GPL License Warning (FFTW)

FFTW3 is licensed under the GNU General Public License (GPL). Distributing binaries
compiled with `POLYMERFTS_USE_FFTW=ON` requires compliance with its GPL terms,
including corresponding source and license notices. Internal/personal use does
not require publishing your code. See the [FFTW FAQ](https://www.fftw.org/faq/section1.html).

## Uninstallation

```bash
python -m pip uninstall polymerfts
```

This removes the Python package while keeping the Conda environment and its
dependencies. Old direct CMake installs have no pip file record; use a fresh
environment when migrating one to wheel installation.
