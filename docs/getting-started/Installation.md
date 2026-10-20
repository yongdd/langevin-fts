# Installation

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

```bash
# Clone repository
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts

# Create and activate conda environment
conda env create -f environment.yml
conda activate polymerfts

# Build and install (enable at least one CPU backend; both are OFF by default)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_MKL=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make -j8
make install
ctest -L basic  # Basic installation verification (~1 minute)
```

`make install` overwrites the modules of an existing install but does not delete
modules that a newer version dropped. When upgrading across versions, remove the old
`polymerfts` directory from `site-packages` first (or use the versioned install below).


## Dependencies

### Required
* **Anaconda**: https://www.anaconda.com/download
* **C++ Compiler**: C++20 support (GCC 10+)
* **spglib**: space group symmetry (`conda install -c conda-forge spglib`, or `apt install libsymspg-dev`)
* **LAPACKE**: Anderson mixing SVD solver (`conda install -c conda-forge liblapacke`, or `apt install liblapacke-dev`; with MKL the header is `mkl_lapacke.h` from `mkl-include`)
* **pybind11 2.13+**, **NumPy 2.0+**, **SciPy 1.14+**: Python bindings (all in `environment.yml`)
* **At least one CPU FFT backend**: Intel MKL (`-DPOLYMERFTS_USE_MKL=ON`) or FFTW3 (`-DPOLYMERFTS_USE_FFTW=ON`)

### Optional
* **CUDA Toolkit 11.8+**: GPU backend (https://developer.nvidia.com/cuda-toolkit)
* **FFTW3**: Alternative CPU backend — enable with `-DPOLYMERFTS_USE_FFTW=ON`

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `POLYMERFTS_USE_CUDA` | ON | Enable NVIDIA CUDA GPU backend |
| `POLYMERFTS_USE_MKL` | OFF | Enable Intel MKL CPU backend |
| `POLYMERFTS_USE_FFTW` | OFF | Enable FFTW3 CPU backend (**GPL license**) |
| `POLYMERFTS_BUILD_TESTS` | ON | Build test executables |
| `POLYMERFTS_INSTALL_PYTHON` | ON | Install Python module |
| `POLYMERFTS_CUDA_ARCHITECTURES` | `60;61;70;75;80;86;89;90` | CUDA compute capabilities to compile for |
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

**Error**: `Unsupported gpu architecture 'compute_89'`

**Cause**: the build compiles for compute capabilities 60-90 by default
(`POLYMERFTS_CUDA_ARCHITECTURES`), and a toolkit may not support all of them
(CUDA 13.x drops sm_60/61; older toolkits lack 89/90).

**Solution**: list only the capabilities your toolkit and GPU support, or update the toolkit:
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

**Solution**: Ensure `nvcc` is in your PATH when configuring (the installed module
records the library directories itself):
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

## Keeping several versions side by side

Python installs one version of a package per environment, so two versions of
`polymerfts` cannot share one `site-packages`. The standard way to keep, say,
the version a paper was computed with next to the current one is a wheel per
version plus a small virtual environment per version:

```bash
conda activate polymerfts                 # provides CMake, CUDA, MKL/FFTW, spglib, LAPACKE
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

Options: `--mkl ON|OFF`, `--fftw ON|OFF`, `--cuda ON|OFF` (default all ON),
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
| **Full** | `ctest` | 93 | ~7 min | Development validation |

**Note**: `ctest` runs the Python tests against the package staged in `build/python/` by `make`, so no `make install` is needed for testing; `make install` is what the examples and tutorials use.

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

**Full tests** add (29 tests):
- SCFT: 3D, hexagonal cylinder
- Stress tensor: 2D, 3D calculations
- Complex field: CL-FTS propagators
- Solvent: polymer-solvent systems
- Space group phases: free energy validation (BCC, Gyroid, etc.)
- Nanoparticle, absorbing boundary conditions
- Complex-field stress on every platform, RK2 stress with non-periodic boundaries
- L-FTS/CL-FTS run loops and checkpoint continuation, convergence orders
