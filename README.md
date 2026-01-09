# Polymer Field Theory Simulations with Python

This repository contains a library for polymer field theory simulations and their applications, such as Self-Consistent Field Theory (SCFT), Langevin Field-Theoretic Simulation (L-FTS), and Complex Langevin FTS (CL-FTS). The most time-consuming and common tasks in polymer field theory simulations are the computation of chain propagators, stresses, partition functions, and polymer concentrations in external fields. These routines are implemented in C++/CUDA and provided as Python classes, enabling you to write programs using Python with numerous useful libraries. This library automatically avoids redundant computations in the chain propagator calculations for branched polymers.

This open-source code is distributed under the Apache License 2.0. This license is one of the permissive software licenses and has minimal restrictions.

## Features

### Core Library
  * Any number of monomer types
  * Arbitrary acyclic branched polymers
  * Arbitrary mixtures of block copolymers and homopolymers
  * Arbitrary initial conditions of propagators at chain ends
  * Access to chain propagators
  * Conformational asymmetry
  * Simulation box dimension: 3D, 2D, and 1D
  * Automatic optimization of chain propagator computations
  * Chain models: continuous, discrete
  * Pseudo-spectral method
    * 4th-order Richardson extrapolation method for continuous chains
    * Supports continuous and discrete chains
    * Periodic boundaries only
  * Real-space method
    * 2nd-order Crank-Nicolson method
    * Supports only continuous chains
    * Supports periodic, reflecting, and absorbing boundaries
  * Can set an impenetrable region using a mask
  * Can constrain space group symmetries during the SCFT iterations (orthorhombic, tetragonal, and cubic only) (**beta**)
  * Anderson mixing
  * Platforms: MKL (CPU) and CUDA (GPU)
  * Parallel computations of propagators with multi-core CPUs (up to 8) or multiple CUDA streams (up to 4) to maximize GPU usage
  * Memory saving option
  * Common interfaces regardless of chain model, simulation box dimension, and platform

### SCFT, L-FTS, and CL-FTS Modules
On top of the above library, SCFT, L-FTS, and CL-FTS are implemented. They support the following features:
  * Polymer melts
  * Arbitrary mixtures of block copolymers, homopolymers, and random copolymers
  * Box size determination by stress calculation (for SCFT)
  * Leimkuhler-Matthews method for updating exchange field (for L-FTS and CL-FTS)
  * Random Number Generator: PCG64 (for L-FTS and CL-FTS)
  * Complex-valued auxiliary fields for handling the sign problem (for CL-FTS)
  * Dynamical stabilization option (for CL-FTS)
  * Smearing for finite-range interactions (for CL-FTS)

## Installation

### Quick Start

#### Option 1: Docker (Recommended for Quick Setup)

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

#### Option 2: Conda Environment + Build from Source

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate polymerfts

# Clone and build
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
make test
make install
```

#### Option 3: pip Install (Requires MKL)

If you have Intel MKL installed:
```bash
pip install .
```

### Dependencies

#### Required
* **C++ Compiler**: Any compiler supporting C++20 standard
* **Intel MKL**: For CPU computations (install via Intel oneAPI toolkit or conda-forge)
* **Python 3.11+**: With NumPy, SciPy, matplotlib, networkx, pyyaml
* **CMake 3.17+**: Build system
* **pybind11**: Python-C++ binding

#### Optional
* **CUDA Toolkit 11.8+**: For GPU computation (https://developer.nvidia.com/cuda-toolkit)
  * Set `CUDA_ARCHITECTURES` in `CMakeLists.txt` for your GPU: https://developer.nvidia.com/cuda-gpus

#### Development Tools (Optional)
* **Anaconda/Miniconda**: https://www.anaconda.com/
* **Visual Studio Code**: https://code.visualstudio.com/
  * Recommended extensions: C/C++, CMake Tools, Remote - SSH, Jupyter, Python

### Build from Source (Detailed)

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

# Run tests
make test

# Install to conda environment
make install
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `POLYMERFTS_USE_MKL` | ON | Enable Intel MKL CPU backend |
| `POLYMERFTS_USE_CUDA` | ON | Enable NVIDIA CUDA GPU backend |
| `POLYMERFTS_BUILD_TESTS` | ON | Build test executables |
| `POLYMERFTS_INSTALL_PYTHON` | ON | Install Python module |

Example with options:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_CUDA=OFF
```

### Troubleshooting

* **Unsupported gpu architecture 'compute_89'**:
  * Remove `;89;90` from `CUDA_ARCHITECTURES` in `CMakeLists.txt`, or
  * Update CUDA Toolkit to a newer version

* **Segmentation fault**:
```bash
ulimit -s unlimited       # Add to ~/.bashrc
export OMP_STACKSIZE=1G   # Stack size for OpenMP
```

* **MKL not found**: Set the MKL root directory:
```bash
export MKLROOT=/path/to/mkl  # or $CONDA_PREFIX for conda-forge MKL
cmake .. -DMKL_ROOT=$MKLROOT
```

### Uninstallation

```bash
# Remove conda environment
conda deactivate
conda env remove -n polymerfts

# Remove Docker images (if used)
docker rmi polymerfts:cpu polymerfts:cuda
```

## User Guide

### Getting Started
+ To use this library, first activate the virtual environment by typing `conda activate polymerfts` in the command line.
+ To learn how to use it, please read the files in the `tutorials` folder.

### Units and Conventions
+ The unit of length in this library is ${bN^{1/2}}$ for both `Continuous` and `Discrete` chain models, where $b$ is a reference statistical segment length and $N$ is a reference polymerization index. The fields acting on chains are defined as `per reference chain` potential instead of `per reference segment` potential. The same notation is used in [*Macromolecules* **2013**, 46, 8037]. If you want to obtain the `per reference segment` potential, multiply $ds$ to each field.

### Performance Tips
+ Set 'reduce_memory_usage=True' if memory is insufficient to run your simulation. However, the execution time increases by several times. The method is based on the idea used in pscfplus (https://github.com/qwcsu/pscfplus/blob/master/doc/notes/SavMem.pdf).
+ The CUDA version also uses multiple CPUs. Each CPU is responsible for a CUDA computation stream. Allocate as many CPUs as `OMP_NUM_THREADS` when submitting a job.
+ To run a simulation using only 1 CPU core, set `os.environ["OMP_MAX_ACTIVE_LEVELS"]="0"` in the Python script.

### Using SCFT, L-FTS, and CL-FTS Modules
+ The `scft.py`, `lfts.py`, and `clfts.py` modules are implemented on top of the `_core` library and `polymer_field_theory.py`.
  + There are examples in `examples/scft`, `examples/lfts`, and `examples/clfts`, respectively.
  + If your SCFT calculation does not converge, set "am.mix_min"=0.01 and "am.mix_init"=0.01, and reduce "am.start_error" in the parameter set.
  + The default platform is cuda for 2D and 3D, and cpu-mkl for 1D.
  + In `lfts.py`, the structure function is computed under the assumption that $\left<w({\bf k})\right>\left<\phi(-{\bf k})\right>$ is zero.
  + In `clfts.py`, the fields are complex-valued and the full FFT is used for structure function calculations.
+ If your ultimate goal is to use deep learning boosted L-FTS, you may use the sample scripts from the DL-FTS repository (https://github.com/yongdd/deep-langevin-fts). One can easily turn on/off deep learning from the scripts.

### Validation
+ Open-source software has no warranty. Make sure that this program reproduces the results of previous SCFT and FTS studies and also produces reasonable results. For acyclic branched polymers adopting the `Continuous` model with an even number of contour steps, the results must be identical to those of PSCF (https://github.com/dmorse/pscfpp) within machine precision. For AB diblock copolymers adopting the `Discrete` model, the results must be identical to those of the code in [*Polymers* **2021**, 13, 2437].
+ It must produce the same results within machine precision regardless of platform (CUDA or MKL), use of superposition, and use of the memory saving option. After changing 'platform' and 'aggregate_propagator_computation', run a few iterations with the same simulation parameters and check if it outputs the same results.

### Additional Tools
+ MATLAB and Python tools for visualization and renormalization are included in the `tools` folder.

## Citation
If you use this software in your research, please cite the following paper:

D. Yong, and J. U. Kim, Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations. *J. Chem. Theory Comput.* **2025**, 21, 3676

## References

#### Discrete Chain Model
+ S. J. Park, D. Yong, Y. Kim, and J. U. Kim, Numerical implementation of pseudo-spectral method in self-consistent mean field theory for discrete polymer chains. *J. Chem. Phys.* **2019**, 150, 234901

#### Multi-Monomer Polymer Field Theory
+ D. Morse, D. Yong, and K. Chen, Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems. *Macromolecules* **2025**, 58, 816

#### CUDA Implementation
+ G. K. Cheong, A. Chawla, D. C. Morse, and K. D. Dorfman, Open-source code for self-consistent field theory calculations of block polymer phase behavior on graphics processing units. *Eur. Phys. J. E* **2020**, 43, 15
+ D. Yong, Y. Kim, S. Jo, D. Y. Ryu, and J. U. Kim, Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces. *Macromolecules* **2021**, 54, 11304

#### Langevin FTS
+ M. W. Matsen, and T. M. Beardsley, Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation. *Polymers* **2021**, 13, 2437

#### Complex Langevin FTS
+ V. Ganesan, and G. H. Fredrickson, Field-theoretic polymer simulations. *Europhys. Lett.* **2001**, 55, 814
+ K. T. Delaney, and G. H. Fredrickson, Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions. *J. Phys. Chem. B* **2016**, 120, 7615
+ J. D. Willis, and M. W. Matsen, Stabilizing complex-Langevin field-theoretic simulations for block copolymer melts. *J. Chem. Phys.* **2024**, 161, 244903

#### Field Update Algorithm for L-FTS
+ B. Vorselaars, Efficient Langevin and Monte Carlo sampling algorithms: the case of field-theoretic simulations. *J. Chem. Phys.* **2023**, 158, 114117

#### Field Update Algorithm for SCFT
+ A. Arora, D. C. Morse, F. S. Bates, and K. D. Dorfman, Accelerating self-consistent field theory of block polymers in a variable unit cell. *J. Chem. Phys.* **2017**, 146, 244902

## Developer Guide

#### Chain Propagator Computation
Please refer to *J. Chem. Theory Comput.* **2025**, 21, 3676.

#### CUDA Programming
`streams` and `cuFFT` are utilized.
https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

#### Platforms
This program is designed to run on different platforms such as MKL and CUDA, and there is a family of classes for each platform. To produce instances of these classes for a given platform, the `abstract factory pattern` is adopted.

#### Python Binding
`pybind11` is utilized to generate Python interfaces for the C++ classes.
https://pybind11.readthedocs.io/en/stable/index.html

## Contributing

+ Most Python scripts implemented with this library are welcome. They could be very simple scripts for specific polymer morphologies or modified versions of `scft.py`, `lfts.py`, etc.
+ Updating the C++/CUDA code yourself is not recommended. If you want to add new features to the C++ code but find it difficult to modify, please send me sample code written in any programming language.
+ Contributions should contain sample results, test code, or desired outputs to check whether they work correctly.
+ They do not have to be optimal or excellent.
+ There should be relevant published literature.
+ Currently, this library is updated without considering compatibility with previous versions. We will keep managing the contributed code so that it can be executed in the updated version.
+ **Contributed code must not contain GPL-licensed code.**
+ Please do not send me exclusive code from your lab. Make sure that it is allowed as open-source.
+ Any suggestions and advice are welcome, but they may not be reflected.
