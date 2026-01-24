# Installation Guide

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document provides detailed instructions for installing the polymer field theory simulation library.

## Quick Start

### Option 1: Docker (Recommended for Quick Setup)

The easiest way to get started is using Docker:

```bash
# CPU-only (MKL backend)
docker pull polymerfts:cpu
docker run -it --rm -v $(pwd):/home/polymerfts/workspace polymerfts:cpu bash

# GPU (CUDA + MKL)
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
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
ctest -L quick  # Quick installation verification (~5 seconds)
make install
```

### Option 3: pip Install (Requires MKL)

If you have Intel MKL installed:
```bash
pip install .
```

## Dependencies

### Required
* **C++ Compiler**: Any compiler supporting C++20 standard
* **Intel MKL**: For CPU computations (install via Intel oneAPI toolkit or conda-forge)
* **Python 3.11+**: With NumPy, SciPy, matplotlib, networkx, pyyaml
* **CMake 3.17+**: Build system
* **pybind11**: Python-C++ binding

### Optional
* **CUDA Toolkit 11.8+**: For GPU computation (https://developer.nvidia.com/cuda-toolkit)
  * Set `CUDA_ARCHITECTURES` in `CMakeLists.txt` for your GPU: https://developer.nvidia.com/cuda-gpus
* **FFTW3**: Alternative CPU backend (GPL license)
  * Install via conda: `conda install -c conda-forge fftw`

### Development Tools (Optional)
* **Anaconda/Miniconda**: https://www.anaconda.com/
* **Visual Studio Code**: https://code.visualstudio.com/
  * Recommended extensions: C/C++, CMake Tools, Remote - SSH, Jupyter, Python

## Build from Source (Detailed)

```bash
# Create virtual environment
conda create -n polymerfts python=3.12 cmake=3.31 pybind11=2.13 \
    numpy=2.2 scipy=1.14 pandas=2.3 matplotlib=3.10 spglib=2.5 \
    make git pip pyyaml mkl mkl-devel mkl-include \
    jupyter networkx pygraphviz pygments plotly nbformat \
    -c conda-forge

# Activate virtual environment
conda activate polymerfts

# Download the source code
git clone https://github.com/yongdd/langevin-fts.git

# Build
cd langevin-fts && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Run quick tests (installation verification)
ctest -L quick

# Install to conda environment
make install
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `POLYMERFTS_USE_MKL` | ON | Enable Intel MKL CPU backend |
| `POLYMERFTS_USE_FFTW` | OFF | Enable FFTW3 CPU backend (**GPL license**) |
| `POLYMERFTS_USE_CUDA` | ON | Enable NVIDIA CUDA GPU backend |
| `POLYMERFTS_BUILD_TESTS` | ON | Build test executables |
| `POLYMERFTS_INSTALL_PYTHON` | ON | Install Python module |

Example with options:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_CUDA=OFF
```

## FFTW Backend (Optional)

FFTW3 provides an alternative CPU backend. Both MKL and FFTW backends support periodic, reflecting, and absorbing boundary conditions.

### Installing FFTW

Install FFTW3 via conda:
```bash
conda install -c conda-forge fftw
```

Then build with FFTW enabled:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_FFTW=ON
make -j8
```

### GPL License Warning

> **⚠️ Important**: FFTW3 is licensed under the **GNU General Public License (GPL)**.
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

**Solution**:
* Remove `;89;90` from `CUDA_ARCHITECTURES` in `CMakeLists.txt`, or
* Update CUDA Toolkit to a newer version

### Segmentation Fault

**Solution**: Set stack size limits:
```bash
ulimit -s unlimited       # Add to ~/.bashrc
export OMP_STACKSIZE=1G   # Stack size for OpenMP
```

### MKL Not Found

**Solution**: Set the MKL root directory:
```bash
export MKLROOT=/path/to/mkl  # or $CONDA_PREFIX for conda-forge MKL
cmake .. -DMKL_ROOT=$MKLROOT
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
| **Quick** | `ctest -L quick` | 8 | ~5 sec | Installation verification |
| **Full** | `ctest` | 75 | ~3 min | Development validation |

**Quick tests** verify core functionality:
- FFT operations (CPU and CUDA)
- Propagator computation (continuous and discrete chains)
- SCFT solver
- Platform initialization

**Full tests** include additional validation:
- All numerical methods (RQM4, RK2, ETDRK4, CN-ADI)
- Stress tensor calculations
- Branched polymer architectures
- Boundary conditions
- Aggregation optimization

## Verifying Installation

After installation, verify everything works:

```bash
# Activate environment
conda activate polymerfts

# Quick test (recommended for users)
cd build
ctest -L quick

# Full test (for development)
ctest

# Run a simple example
cd ../examples/scft
python Lamella3D.py
```
