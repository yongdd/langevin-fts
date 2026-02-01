# Installation

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

```bash
# Clone repository
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts

# Create and activate conda environment
conda env create -f environment.yml
conda activate polymerfts

# Build and install
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
make install
ctest -L basic  # Basic installation verification (~40 seconds)
```


## Dependencies

### Required
* **Anaconda**: https://www.anaconda.com/download
* **C++ Compiler**: C++20 support (GCC 10+)

### Optional
* **CUDA Toolkit 11.8+**: GPU backend (https://developer.nvidia.com/cuda-toolkit)
* **FFTW3**: Alternative CPU backend — enable with `-DPOLYMERFTS_USE_FFTW=ON`

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `POLYMERFTS_USE_CUDA` | ON | Enable NVIDIA CUDA GPU backend |
| `POLYMERFTS_USE_FFTW` | OFF | Enable FFTW3 CPU backend (**GPL license**) |
| `POLYMERFTS_BUILD_TESTS` | ON | Build test executables |
| `POLYMERFTS_INSTALL_PYTHON` | ON | Install Python module |

### CPU Backends

Two CPU backends are available:
- **MKL** (default): Intel Math Kernel Library, included via conda
- **FFTW**: Enable with `-DPOLYMERFTS_USE_FFTW=ON` (GPL license)

Example (enable FFTW CPU backend):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_FFTW=ON
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

## Uninstallation 😢

```bash
conda deactivate
conda env remove -n polymerfts
```

## Testing

Two test modes are available:

| Mode | Command | Tests | Time | Purpose |
|------|---------|-------|------|---------|
| **Basic** | `ctest -L basic` | 56 | ~40 sec | Installation verification |
| **Full** | `ctest` | 84 | ~4 min | Development validation |

**Note**: Run `make install` before `ctest` so Python tests can import the installed module.

**Basic tests** (56 tests):
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
