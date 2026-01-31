# Installation Guide

> **Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document provides detailed instructions for installing the polymer field theory simulation library.

## Quick Start

### Option 1: Docker (Recommended for Quick Setup)

The easiest way to get started is using Docker:

```bash
# CPU-only (FFTW backend)
docker pull polymerfts:cpu
docker run -it --rm -v $(pwd):/home/polymerfts/workspace polymerfts:cpu bash

# GPU (CUDA + FFTW)
docker pull polymerfts:cuda
docker run -it --rm --gpus all -v $(pwd):/home/polymerfts/workspace polymerfts:cuda bash
```

Or build the Docker images locally:
```bash
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts

# Build CPU image
docker build -f docker/Dockerfile.cpu -t polymerfts:cpu .

# Build GPU image (requires NVIDIA Container Toolkit)
docker build -f docker/Dockerfile.cuda -t polymerfts:cuda .
```

### Option 2: Conda Environment + Build from Source

```bash
# Clone repository
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts

# Create and activate conda environment
conda env create -f environment.yml
conda activate polymerfts

# Build and install
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_FFTW=ON
make -j8
make install
ctest -L basic  # Basic installation verification (~40 seconds)
```

### Option 3: pip Install

Default build (uses CMake defaults: CUDA ON, FFTW OFF):
```bash
pip install .
```


## Dependencies

### Required
* **C++ Compiler**: Any compiler supporting C++20 standard
* **Python 3.11+**: With NumPy, SciPy, matplotlib, networkx, pyyaml
* **CMake 3.17+**: Build system
* **pybind11**: Python-C++ binding

### Optional
* **FFTW3**: CPU backend (default OFF) — enable with `-DPOLYMERFTS_USE_FFTW=ON`
* **CUDA Toolkit 11.8+**: For GPU computation (https://developer.nvidia.com/cuda-toolkit)
  * Set `CUDA_ARCHITECTURES` in `CMakeLists.txt` for your GPU: https://developer.nvidia.com/cuda-gpus

### Development Tools (Optional)
* **Anaconda/Miniconda**: https://www.anaconda.com/
* **Visual Studio Code**: https://code.visualstudio.com/
  * Recommended extensions: C/C++, CMake Tools, Remote - SSH, Jupyter, Python

## Build from Source (Detailed)

```bash
# Create virtual environment
conda create -n polymerfts python=3.12 cmake=3.31 pybind11=2.13 \
    numpy=2.2 scipy=1.14 pandas=2.3 matplotlib=3.10 spglib=2.5 \
    make git pip pyyaml fftw \
    jupyter networkx pygraphviz pygments plotly nbformat \
    -c conda-forge

# Activate virtual environment
conda activate polymerfts

# Download the source code
git clone https://github.com/yongdd/langevin-fts.git

# Build
cd langevin-fts && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_FFTW=ON
make -j8

# Install to conda environment
make install

# Run basic tests (installation verification)
ctest -L basic
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `POLYMERFTS_USE_FFTW` | OFF | Enable FFTW3 CPU backend (**GPL license**) |
| `POLYMERFTS_USE_CUDA` | ON | Enable NVIDIA CUDA GPU backend |
| `POLYMERFTS_BUILD_TESTS` | ON | Build test executables |
| `POLYMERFTS_INSTALL_PYTHON` | ON | Install Python module |

Example with options:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_FFTW=ON -DPOLYMERFTS_USE_CUDA=OFF
```

## GPL License Warning

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

### FFTW Not Found

**Solution**: Install FFTW3 via conda:
```bash
conda install -c conda-forge fftw
```

Or set the FFTW root directory:
```bash
export FFTW_ROOT=/path/to/fftw
cmake .. -DFFTW_ROOT=$FFTW_ROOT
```

### CUDA Not Detected

**Solution**: Ensure CUDA is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### TestStressLinear3D Takes Too Long

**This is normal.** `TestStressLinear3D` takes about 80 seconds to complete because it tests multiple platforms, chain models, and numerical methods with a 3D grid.

## Uninstallation

```bash
# Remove conda environment
conda deactivate
conda env remove -n polymerfts

# Remove Docker images (if used)
docker rmi polymerfts:cpu polymerfts:cuda
```

## Testing

Two test modes are available:

| Mode | Command | Tests | Time | Purpose |
|------|---------|-------|------|---------|
| **Basic** | `ctest -L basic` | ~50 | ~40 sec | Installation verification |
| **Full** | `ctest` | ~65 | ~3 min | Development validation |

**Note**: Run `make install` before `ctest` so Python tests can import the installed module.

**Basic tests** verify core functionality:
- FFT operations (CPU and CUDA)
- Propagator computation (continuous and discrete chains)
- SCFT solver
- Platform initialization

**Full tests** include additional validation:
- All numerical methods (RQM4, RK2, CN-ADI2)
- Stress tensor calculations
- Branched polymer architectures
- Boundary conditions
- Aggregation optimization

## Verifying Installation

After installation, verify everything works:

```bash
# Activate environment
conda activate polymerfts

# Basic test (recommended for users)
cd build
make install
ctest -L basic

# Full test (for development)
ctest

# Run a simple example
cd ../examples/scft
python Lamella3D.py
```
