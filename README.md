# Polymer Field Theory Simulations with Python

# Features
This repository contains a library for polymer field theory simulations and its applications, such as SCFT, L-FTS and CL-FTS. The most time-consuming and common tasks in polymer field theory simulations are the computation of chain propagators, stresses, partition functions, and polymer concentrations in external fields. These routines are implemented in C++/CUDA and provided as Python classes, enabling you to write programs using Python with numerous useful libraries. This library automatically avoids redundant computations in the chain propagator calculations for branched polymers. It supports the following features:
  * Any number of monomer types
  * Arbitrary acyclic branched polymers
  * Arbitrary mixtures of block copolymers and homopolymers
  * Arbitrary initial conditions of propagators at chain ends
  * Access to chain propagators
  * Conformational asymmetry
  * Simulation box dimension: 3D, 2D and 1D
  * Automatic optimization of chain propagator computations
  * Chain models: continuous, discrete
  * Pseudo-spectral method
    * 4th-order Richardson extrapolation method for continuous chain
    * Support continuous and discrete chains
    * Periodic boundaries only
  * Real-space method (**beta**)
    * 2th-order Crank-Nicolson method
    * Support only continuous chain
    * Support periodic, reflecting, absorbing boundaries
  * Can set impenetrable region using a mask (**beta**)
  * Anderson mixing
  * Platforms: MKL (CPU) and CUDA (GPU)
  * Parallel computations of propagators with multi-core CPUs (up to 8), or multi CUDA streams (up to 4) to maximize GPU usage
  * GPU memory saving option
  * Common interfaces regardless of chain model, simulation box dimension, and platform

On the top the above library, SCFT and L-FTS are implemented. They support following features:
  * Polymer melts
  * Arbitrary mixtures of block copolymers, homopolymers, and random copolymer
  * Box size determination by stress calculation (for SCFT)
  * Leimkuhler-Matthews method for updating exchange field (for L-FTS)
  * Random Number Generator: PCG64 (for L-FTS)

This open-source code is distributed under the Apache license 2.0. This license is one of the permissive software licenses and has minimal restrictions.

# Dependencies
#### Linux System

#### C++ Compiler
  Any C++ compiler that supports C++17 standard or higher. To use MKL, install Intel oneAPI toolkit (without Intel Distribution for Python).

#### CUDA Toolkit
  https://developer.nvidia.com/cuda-toolkit   
  It requires CUDA Toolkit Version 11.2 or higher for the GPU computation. If it is not installed, ask admin for its installation.

#### Anaconda
  https://www.anaconda.com/

* * *
Environment variables must be set so that `nvcc` and `conda` can be executed in the command line (Type `which nvcc` and `which conda` to check the installation).

# Installation
```Shell
# Create virtual environment 
conda create -n polymerfts python=3.9 cmake=3.31 pybind11=2.13 \
    make conda git pip scipy openmpi matplotlib pyyaml networkx pygraphviz
# Activate virtual environment  
conda activate polymerfts  
# Download the source code
git clone https://github.com/yongdd/langevin-fts.git  
# Build  
cd langevin-fts && mkdir build && cd build  
cmake ../ -DCMAKE_BUILD_TYPE=Release   
make -j8  
# Run Test  
make test   
# Install  
make install   
```
* **If you encounter `segmentation fault`, type following commands.**     
```Shell
ulimit -s unlimited       # Add this command in ~/.bashrc
export OMP_STACKSIZE=1G   # Stack size for OpenMP
```
*  If you want to remove all installations :cry:, type following commands.   
```Shell
conda deactivate  
conda env remove -n polymerfts  
```
# User Guide
+ To use this library, first activate virtual environment by typing `conda activate polymerfts` in command line. In Python script, import the package by adding  `from polymerfts import *`. To learn how to use it, please see 'examples/ComputeConcentration.py'.
+ Even CUDA version use multiple CPUs. Each of them is responsible for each CUDA computation stream. Allocate multiple CPUs as much as `OMP_NUM_THREADS` when submitting a job.
+ The SCFT and L-FTS are implemented on the python shared library in `examples/scft` and `examples/fts`, respectively.
  + Set 'reduce_gpu_memory_usage=True' if GPU memory space is insufficient to run your simulation. Instead, performance is reduced by 10 ~ 65% depending on chain model and box size.
  + Set 'aggregate_propagator_computation=False, (default: True) if you want to use 'solver.get_block_concentration()', which returns block-wise concentrations of a selected polymer species, and 'solver.get_chain_propagator()', which returns a propagator of a selected branch.
  + If your SCFT calculation does not converge, set "am.mix_min"=0.01 and "am.mix_init"=0.01, and reduce "am.start_error" in parameter set.
  + The default platform is cuda for 2D and 3D, and cpu-mkl for 1D.
  + Use FTS in 1D and 2D only for the tests. It does not have a physical meaning.
  + To run simulation using only 1 CPU core, set `os.environ["OMP_MAX_ACTIVE_LEVELS"]="0"` in the python script. As an example, please see 'examples/scft/Gyroid.py'.
  + The structure function is computed under the assumption that <w(k)><phi(-k)> is zero.
  + The Hamiltonian and free energy in `examples/scft` and `examples/fts` are defined as *βH × R_0^3/(V√N)* and *βF × R_0^3/(V√N)*, respectively. They become per chain expressions only when `volume_fraction` and `alpha` are 1.
  + L-FTS is one of the partial saddle-point approximation methods, applying the saddle-point approximation to all imaginary fields. L-FTS is known to give accurate results for AB-type block copolymers, where the saddle-point approximation is applied only to the pressure field. However, it has not been confirmed whether L-FTS yields accurate results for ABC-type block copolymers, where one of the exchange fields can be an imaginary field. In this case, field fluctuations may not be fully captured. Use L-FTS in this situation at your own risk (see G. H. Fredrickson and K. T. Delaney, **2023**, Oxford).
+ If your ultimate goal is to use deep learning boosted L-FTS, you may use the sample scripts of DL-FTS repository. (https://github.com/yongdd/deep-langevin-fts) (One can easily turn on/off deep learning from the scripts.)
+ The unit of length in this library is *aN^(1/2)* for both `Continuous` and `Discrete` chain models, where *a* is a reference statistical segment length and *N* is a reference polymerization index. The fields acting on chain are defined as `per reference chain` potential instead of `per reference segment` potential. The same notation is used in [*Macromolecules* **2013**, 46, 8037]. If you want to obtain the same fields used in [*Polymers* **2021**, 13, 2437], multiply *ds* to each field. Please refer to [*J. Chem. Phys.* **2014**, 141, 174103] to learn how to formulate polymer mixtures composed of multiple distinct polymers in the reference polymer length unit.
+ Open-source has no warranty. Make sure that this program reproduces the results of previous SCFT and FTS studies, and also produces reasonable results. For acyclic branched polymers adopting the `Continuous` model with an even number of contour steps, the results must be identical to those of PSCF (https://github.com/dmorse/pscfpp) within the machine precision. For AB diblock copolymers adopting the `Discrete` model, the results must be identical to those of code in [*Polymers* **2021**, 13, 2437].
+ It must produce the same results within the machine precision regardless of platform (CUDA or MKL), use of superposition, and use of GPU memory saving option. After changing 'platform' and 'aggregate_propagator_computation', run a few iterations with the same simulation parameters. And check if it outputs the same results.

+ Matlab and Python tools for visualization and renormalization are included in `tools` folder.

# Contribution
+ Most of python scripts implemented with this library are welcome. They could be very simple scripts for specific polymer morphologies, or modified versions of `scft.py`, `lfts.py`, etc.
+ Updating C++/CUDA codes by yourself is not recommended. If you want to add new features to C++ part but it is hard to modify it, please send me sample codes written in any programming languages.
+ They should contain sample results, test codes, or desired outputs to check whether they work correctly.
+ They do not have to be optimal or to be excellent.
+ There should be relevant published literatures.
+ Currently, this library is updated without considering compatibility with previous versions. We will keep managing the contributed codes so that they can be executed in the updated version.
+ **Contributed codes must not contains GPL-licensed codes.**
+ Please do not send me exclusive codes of your lab. Make sure that they are allowed as open-source.
+ Any suggestions and advices are welcome, but they could not be reflected.

# Developer Guide

#### Chain Propagator Computation
  Please refer to *J. Chem. Theory Comput.* **2025**, 21, 3676.

#### Reducing GPU Memory Usage
  1. Propagators of all segments are stored in the GPU's global memory to minimize data transfer between main memory and global memory, because data transfer operations are expensive. However, this method limits the sizes of the grid number and segment number. If the GPU memory space is not enough to run simulations, the propagators should be stored in main memory instead of GPU memory. To reduce data transfer time, `device overlap` can be utilized, which simultaneously transfers data and executes kernels. An example applied to AB diblock copolymers is provided in the supporting information of [*Macromolecules* **2021**, 54, 11304]. To enable this option, set 'reduce_gpu_memory_usage' to 'True' in the example script. If this option is enabled, the factory will create an instance of CudaComputationReduceMemoryDiscrete or CudaComputationReduceMemoryDiscrete.
  2. In addition, when 'reduce_gpu_memory_usage' is enabled, field history for Anderson Mixing is also stored in main memory, and the factory will create CudaAndersonMixingReduceMemory.

#### Platforms  
  This program is designed to run on different platforms such as MKL and CUDA, and there is a family of classes for each platform. To produce instances of these classes for given platform, `abstract factory pattern` is adopted.   

#### Python Binding  
  `pybind11` is utilized to generate Python interfaces for the C++ classes.  
  https://pybind11.readthedocs.io/en/stable/index.html   

# References
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

# Citation
D. Yong, and J. U. Kim, Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations. *J. Chem. Theory Comput.* **2025**, 21, 3676