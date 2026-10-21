# Installation

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

```bash
# Clone repository
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts

# Create and activate conda environment
conda env create -f environment.yml
conda activate polymerfts

# GPU backend (optional): a CUDA 12.2+ nvcc first on PATH; the conda environment has none
export PATH=/usr/local/cuda-12.4/bin:$PATH

# Build and install (enable at least one CPU backend; both are OFF by default)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_MKL=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make -j8
make install
ctest -L basic  # Basic installation verification (~1 minute)

# Platforms that were built (cuda appears only with a usable CUDA toolchain and a GPU)
python -c "from polymerfts import PlatformSelector; print(PlatformSelector.avail_platforms())"
```

The configure step prints `CUDA toolkit ...` when it builds the GPU backend, and a
`Building without the GPU backend: ...` warning with the reason when it does not
(see [CUDA Toolkit Too Old](#cuda-toolkit-too-old)).

`make install` overwrites the modules of an existing install but does not delete
modules that a newer version dropped. When upgrading across versions, remove the old
`polymerfts` directory from `site-packages` first (or use the versioned install below).


## Dependencies

### Required
* **Anaconda**: https://www.anaconda.com/download
* **Python 3.11+** (`environment.yml` installs 3.12)
* **CMake 3.17+** (3.25.2+ for the GPU backend; `environment.yml` installs 3.31)
* **C++ Compiler**: C++20 support (GCC 10+; GCC 11+ or Clang 15+ for the GPU backend, which uses `std::source_location`)
* **spglib**: space group symmetry (`conda install -c conda-forge spglib`, or `apt install libsymspg-dev`)
* **LAPACKE**: Anderson mixing SVD solver (`conda install -c conda-forge liblapacke`, or `apt install liblapacke-dev`; with MKL the header is `mkl_lapacke.h` from `mkl-include`)
* **pybind11 2.13+**, **NumPy 2.0+**, **SciPy 1.14+**: Python bindings (all in `environment.yml`)
* **At least one CPU FFT backend**: Intel MKL (`-DPOLYMERFTS_USE_MKL=ON`) or FFTW3 (`-DPOLYMERFTS_USE_FFTW=ON`)

### Optional
* **CUDA Toolkit 12.2+**: GPU backend (https://developer.nvidia.com/cuda-toolkit). The device code is C++20, which nvcc 11.x cannot compile; nvcc 12.0/12.1 cannot compile `std::source_location`. The GPU backend also needs CMake 3.25.2+ (`environment.yml` installs 3.31), an NVIDIA driver for CUDA 12 (R525 or newer), and a host compiler the toolkit supports (nvcc 12.2 rejects GCC 13, nvcc 12.4 accepts it)
* **FFTW3**: Alternative CPU backend — enable with `-DPOLYMERFTS_USE_FFTW=ON`

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `POLYMERFTS_USE_CUDA` | AUTO | NVIDIA CUDA GPU backend. `AUTO`: build it when a usable toolchain is found (CUDA 12.2+, CMake 3.25.2+, `std::source_location` in the compilers), otherwise warn and build without it; `ON`: require it (configure stops otherwise); `OFF`: never |
| `POLYMERFTS_USE_MKL` | OFF | Enable Intel MKL CPU backend |
| `POLYMERFTS_USE_FFTW` | OFF | Enable FFTW3 CPU backend (**GPL license**) |
| `POLYMERFTS_BUILD_TESTS` | ON | Build test executables |
| `POLYMERFTS_INSTALL_PYTHON` | ON | Install Python module |
| `POLYMERFTS_CUDA_ARCHITECTURES` | empty | CUDA compute capabilities to compile for; empty: `60;61;70;75;80;86;89;90` with CUDA 12, `75;80;86;89;90;100;120` with CUDA 13+ |
| `POLYMERFTS_WERROR` | OFF | Treat host C++ warnings as errors (CI builds with it) |
| `POLYMERFTS_INSTALL_HEADERS` | OFF | Install C++ headers and a CMake package config for downstream projects (experimental) |

### CPU Backends

Two CPU backends are available, and **both are OFF by default** — enable at
least one (or build CUDA-only):
- **MKL**: Intel Math Kernel Library, included via conda — enable with `-DPOLYMERFTS_USE_MKL=ON`
- **FFTW**: enable with `-DPOLYMERFTS_USE_FFTW=ON` (GPL license)

Example (enable the MKL CPU backend):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_MKL=ON
```

## GPL License Warning (FFTW)

> **Important**: FFTW3 is licensed under the **GNU General Public License (GPL)**.
>
> If you distribute binaries compiled with `POLYMERFTS_USE_FFTW=ON`, you **must** comply with GPL terms:
> 1. Distribute the complete source code of your application
> 2. License your application under GPL or a GPL-compatible license
> 3. Include the full GPL license text with your distribution
>
> For internal/personal use, no action is required.
>
> See: https://www.fftw.org/faq/section1.html

## Troubleshooting

### Unsupported GPU Architecture

**Error**: `Unsupported gpu architecture 'compute_60'` (or another capability)

**Cause**: `POLYMERFTS_CUDA_ARCHITECTURES` lists a capability the toolkit does not
support. The default list follows the toolkit (CUDA 13 dropped sm_60, sm_61 and
sm_70), but a build directory configured earlier keeps the list it was given.

**Solution**: list only the capabilities your toolkit and GPU support (or set the
option to an empty string to get the default for the toolkit):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_CUDA_ARCHITECTURES="80;86"
```

### Segmentation Fault

**Solution**: Set stack size limits:
```bash
ulimit -s unlimited       # Add to ~/.bashrc
export OMP_STACKSIZE=1G   # Stack size for OpenMP
```

### CUDA Not Detected

**Message**: `Building without the GPU backend: no working CUDA compiler (nvcc) was found`
(a warning with the default `POLYMERFTS_USE_CUDA=AUTO`), and `cuda` is missing from
`PlatformSelector.avail_platforms()`.

**Solution**: put a CUDA 12.2+ `nvcc` on PATH (or pass `-DCMAKE_CUDA_COMPILER=...`) and
configure a new build directory; the installed module records the library directories
itself, so `LD_LIBRARY_PATH` is not needed at run time:
```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
```
`cuda` is also missing at run time on a machine without a visible GPU (for example a CPU
node of a cluster), even when the GPU backend was built.

### CUDA Toolkit Too Old

**Message**: `Building without the GPU backend: .../nvcc is CUDA 11.8; the GPU backend
needs CUDA 12.2 or newer` (a warning with the default `POLYMERFTS_USE_CUDA=AUTO`; with
`-DPOLYMERFTS_USE_CUDA=ON` the configure step stops with the same text). CMake older
than 3.25.2 and a compiler without `std::source_location` (GCC 10, Clang 14) are
reported the same way.

**Solution**: put a CUDA 12.2+ toolkit first on PATH before configuring a new build
directory (an existing build directory keeps the compiler it was configured with, so
configure a fresh one or delete its `CMakeCache.txt`):
```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_MKL=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
         -DPOLYMERFTS_USE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc
```
A conda environment may also hold CUDA runtime libraries of another version (for
example `cudatoolkit` installed for CuPy). The build links the cuFFT of the toolkit
whose `nvcc` compiles the code; the configure step prints the path it uses (`cuFFT: ...`).

## Keeping several versions side by side

Python installs one version of a package per environment, so two versions of
`polymerfts` cannot share one `site-packages`. The standard way to keep, say,
the version a paper was computed with next to the current one is a wheel per
version plus a small virtual environment per version:

```bash
conda activate polymerfts                 # provides CMake, MKL/FFTW, spglib, LAPACKE
export PATH=/usr/local/cuda-12.4/bin:$PATH  # a CUDA 12.2+ nvcc for the GPU backend (not in conda)
scripts/install_versioned.sh              # builds ~/wheels/polymerfts-<version>-*.whl and
                                          # installs it into ~/venvs/polymerfts-<version>
```

The venv is created with `--system-site-packages`, so NumPy, SciPy and the
other dependencies are shared with the conda environment and the venv holds
only `polymerfts` (a few tens of MB). Select a version per job:

```bash
source ~/venvs/polymerfts-0.10.0/bin/activate   # in an sbatch script: this line before python
python Gyroid.py
```

Options: `--mkl ON|OFF`, `--fftw ON|OFF`, `--cuda AUTO|ON|OFF` (default all ON; with
`--cuda ON` a missing CUDA 12.2+ toolchain stops the build, `AUTO` builds without it),
`-j N` build parallelism (default 4), `--wheel-dir`, `--venv-dir`,
`--skip-build` to reinstall an already built wheel, `--no-build-isolation`
when working offline (then `scikit-build-core`, `pybind11` and `numpy` must be
installed in the conda environment). To keep an older release, check it out in
a separate work tree and run the script from there (releases from 0.10.0 on;
older tags ship no `pyproject.toml`, so build them with `make install` from
their own build directory instead):

```bash
git worktree add ../langevin-fts-0.10.0 v0.10.0
(cd ../langevin-fts-0.10.0 && scripts/install_versioned.sh)
```

For quick comparisons during development, the package staged in
`build/python` after every `make` can be used without installing anything:
`PYTHONPATH=/path/to/build/python python script.py`.

## Uninstallation 😢

```bash
conda deactivate
conda env remove -n polymerfts
```

## Testing

Two test modes are available:

| Mode | Command | Tests | Time | Purpose |
|------|---------|-------|------|---------|
| **Basic** | `ctest -L basic` | 57 | ~1 min | Installation verification |
| **Full** | `ctest` | 94 | ~7 min | Development validation |

**Note**: `ctest` runs the Python tests against the package staged in `build/python/` by `make`, so no `make install` is needed for testing; `make install` is what the examples and tutorials use.

The counts are for a build with the GPU backend and both CPU backends; the tests of a
backend that is not built are left out.

**Basic tests** (57 tests):
- FFT: CPU/CUDA, DCT, DST, mixed boundary conditions
- Propagator: linear, branched, mixture polymers
- Numerical methods: pseudo-spectral (RQM4, RK2), real-space (CN-ADI)
- SCFT: 1D, 2D solvers
- Stress tensor: 1D calculations
- Space group: reduced basis implementation
- CrysFFT: crystallographic FFT
- Aggregation: propagator optimization
- Scheduler: parallel computation

**Full tests** add (37 tests):
- SCFT: 3D, hexagonal cylinder
- Stress tensor: 2D, 3D calculations
- Complex field: CL-FTS propagators
- Solvent: polymer-solvent systems
- Space group phases: free energy validation (BCC, Gyroid, etc.)
- Nanoparticle, absorbing boundary conditions
- Complex-field stress on every platform, RK2 stress with non-periodic boundaries
- L-FTS/CL-FTS run loops and checkpoint continuation, convergence orders
- CUDA memory ownership: every object returns its device and pinned memory, also when an allocation fails or the device runs out of memory
