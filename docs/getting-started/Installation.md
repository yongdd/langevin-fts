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
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_MKL=ON
make -j8
make install
ctest -L basic  # Basic installation verification (~1 minute)
```


## Dependencies

### Required
* **Anaconda**: https://www.anaconda.com/download
* **C++ Compiler**: C++20 support (GCC 10+)
* **spglib**: space group symmetry (`conda install -c conda-forge spglib`, or `apt install libsymspg-dev`)
* **LAPACKE**: Anderson mixing SVD solver (`conda install -c conda-forge lapack`, or `apt install liblapacke-dev`)
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

**Cause**:
`CMakeLists.txt` sets:
`CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90"`.
Newer CUDA toolkits may not support some of these (e.g., CUDA 13.x drops sm_60/61),
and you may also see `compute_89/90` errors if those are not supported by your toolkit.

**Solution**:
* Remove unsupported compute capabilities (e.g., 89/90 or 60/61) from `CUDA_ARCHITECTURES`, or
* Set `CMAKE_CUDA_ARCHITECTURES` explicitly for your GPU, or
* Update CUDA Toolkit to a newer version

Examples:
```bash
# Configure with an explicit architecture
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
```
Or edit `CMakeLists.txt` to set `CUDA_ARCHITECTURES` to a supported value.

### Segmentation Fault

**Solution**: Set stack size limits:
```bash
ulimit -s unlimited       # Add to ~/.bashrc
export OMP_STACKSIZE=1G   # Stack size for OpenMP
```

### CUDA Not Detected

**Solution**: Ensure CUDA is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
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
a separate work tree and run the script from there:

```bash
git worktree add ../langevin-fts-0.9.1 v0.9.1
(cd ../langevin-fts-0.9.1 && scripts/install_versioned.sh)
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
| **Basic** | `ctest -L basic` | ~57 | ~1 min | Installation verification |
| **Full** | `ctest` | ~82 | ~6 min | Development validation |

**Note**: `ctest` runs the Python tests against the package staged in `build/python/` by `make`, so no `make install` is needed for testing; `make install` is what the examples and tutorials use.

**Basic tests** (~57 tests):
- FFT: CPU/CUDA, DCT, DST, mixed boundary conditions
- Propagator: linear, branched, mixture polymers
- Numerical methods: pseudo-spectral (RQM4, RK2), real-space (CN-ADI)
- SCFT: 1D, 2D solvers
- Stress tensor: 1D calculations
- Space group: reduced basis implementation
- CrysFFT: crystallographic FFT
- Aggregation: propagator optimization
- Scheduler: parallel computation

**Full tests** add (28 tests):
- SCFT: 3D, hexagonal cylinder
- Stress tensor: 2D, 3D calculations
- Complex field: CL-FTS propagators
- Solvent: polymer-solvent systems
- Space group phases: free energy validation (BCC, Gyroid, etc.)
- Nanoparticle, absorbing boundary conditions
