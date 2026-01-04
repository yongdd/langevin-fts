# Polymer Field Theory Simulations with Python

This repository contains a library for polymer field theory simulations and their applications, such as SCFT and L-FTS. The most time-consuming and common tasks in polymer field theory simulations are the computation of chain propagators, stresses, partition functions, and polymer concentrations in external fields. These routines are implemented in C++/CUDA and provided as Python classes, enabling you to write programs using Python with numerous useful libraries. This library automatically avoids redundant computations in the chain propagator calculations for branched polymers.

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
  * Real-space method (**beta**)
    * 2nd-order Crank-Nicolson method
    * Supports only continuous chains
    * Supports periodic, reflecting, and absorbing boundaries
  * Can set an impenetrable region using a mask (**beta**)
  * Can constrain space group symmetries during the SCFT iterations (orthorhombic, tetragonal, and cubic only) (**beta**)
  * Anderson mixing
  * Platforms: MKL (CPU) and CUDA (GPU)
  * Parallel computations of propagators with multi-core CPUs (up to 8) or multiple CUDA streams (up to 4) to maximize GPU usage
  * Memory saving option
  * Common interfaces regardless of chain model, simulation box dimension, and platform

### SCFT and L-FTS Modules
On top of the above library, SCFT and L-FTS are implemented. They support the following features:
  * Polymer melts
  * Arbitrary mixtures of block copolymers, homopolymers, and random copolymers
  * Box size determination by stress calculation (for SCFT)
  * Leimkuhler-Matthews method for updating exchange field (for L-FTS)
  * Random Number Generator: PCG64 (for L-FTS)

## Installation

### Dependencies

#### Linux System

#### C++ Compiler
Any C++ compiler that supports the C++20 standard or higher. To use MKL, install the Intel oneAPI toolkit (without Intel Distribution for Python).

#### CUDA Toolkit
https://developer.nvidia.com/cuda-toolkit

CUDA Toolkit version 11.8 or higher is required for GPU computation. If it is not installed, ask your system administrator for assistance.

You need to set `CUDA_ARCHITECTURES` in `CMakeLists.txt` depending on your GPU system.
https://developer.nvidia.com/cuda-gpus

#### Anaconda
https://www.anaconda.com/

#### (Optional) Visual Studio Code
https://code.visualstudio.com/

Install the following extensions in the 'Extensions' tab:
  * C/C++
  * CMake Tools
  * Remote - SSH
  * Jupyter
  * Python

---
Environment variables must be set so that `nvcc` and `conda` can be executed in the command line (type `which nvcc` and `which conda` to check the installation).

### Build Instructions
```Shell
# Create virtual environment
conda create -n polymerfts python=3.9 cmake=3.31 pybind11=2.13 \
    make conda git pip scipy openmpi matplotlib pyyaml \
    jupyter networkx pygraphviz pygments plotly nbformat
# Activate virtual environment
conda activate polymerfts
# Install spglib for the space group
pip install spglib
# Download the source code
git clone https://github.com/yongdd/langevin-fts.git
# Build
cd langevin-fts && mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
make -j8
# Run tests
make test
# Install
make install
```

### Troubleshooting
* **If you encounter `Unsupported gpu architecture 'compute_89'`, try one of the following:**
  * Remove `;89;90` from `CUDA_ARCHITECTURES` in `CMakeLists.txt`.
  * Update CUDA Toolkit.
* **If you encounter a `segmentation fault`, type the following commands:**
```Shell
ulimit -s unlimited       # Add this command to ~/.bashrc
export OMP_STACKSIZE=1G   # Stack size for OpenMP
```

### Uninstallation
If you want to remove all installations :cry:, type the following commands:
```Shell
conda deactivate
conda env remove -n polymerfts
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

### Using SCFT and L-FTS Modules
+ The `scft.py` and `lfts.py` modules are implemented on top of the `_core` library and `polymer_field_theory.py`.
  + There are examples in `examples/scft` and `examples/fts`, respectively.
  + If your SCFT calculation does not converge, set "am.mix_min"=0.01 and "am.mix_init"=0.01, and reduce "am.start_error" in the parameter set.
  + The default platform is cuda for 2D and 3D, and cpu-mkl for 1D.
  + In `lfts.py`, the structure function is computed under the assumption that $\left<w({\bf k})\right>\left<\phi(-{\bf k})\right>$ is zero.
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

#### Multi-Monomer Polymer Field Theory
+ D. Morse, D. Yong, and K. Chen, Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems. *Macromolecules* **2025**, 58, 816

#### CUDA Implementation
+ G. K. Cheong, A. Chawla, D. C. Morse, and K. D. Dorfman, Open-source code for self-consistent field theory calculations of block polymer phase behavior on graphics processing units. *Eur. Phys. J. E* **2020**, 43, 15
+ D. Yong, Y. Kim, S. Jo, D. Y. Ryu, and J. U. Kim, Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces. *Macromolecules* **2021**, 54, 11304

#### Langevin FTS
+ M. W. Matsen, and T. M. Beardsley, Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation. *Polymers* **2021**, 13, 2437

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
