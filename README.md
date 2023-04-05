# Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) for Python

# Features
This is not just an application, but it contains a library for polymer field theory simulations. The most time-consuming and common routines in polymer field theory simulations are the computation of stresses, partition functions and concentrations of polymers in external fields. These routines are written in C++/CUDA and provided as python classes in this library. With these classes, you can write your own programs using python language, and your applications can be easily customized and extended by adopting numerous useful python libraries. Moreover, it automatically optimize the computation of chain propagators for any mixture of arbitrary acyclic branched polymers. You can focus on development of your applications rather than tedious optimization for branched polymers. This library supports following features:
  * Any number of monomer types
  * Arbitrary acyclic branched polymers (**beta**)
  * Arbitrary mixtures of block copolymers and homopolymers (**beta**)
  * Arbitrary initial conditions of propagators at chain ends (**beta**)
  * Conformational asymmetry
  * Chain models: continuous, discrete
  * Simulation box dimension: 3D, 2D and 1D
  * Periodic boundaries
  * Automatic optimization to compute chain propagators with minimal iterations (**beta**)
  * Pseudospectral method (4th-order method for continuous chain)
  * Anderson mixing
  * Platforms: MKL (CPU) and CUDA (GPU)
  * GPU memory saving option (**beta**)
  * Parallel computations of propagators with multi-core CPUs, batched cuFFT, or two GPUs (**beta**)
  * Common interfaces regardless of chain model, simulation box dimension, and platform

Using the above python shared library, SCFT and L-FTS are implemented. They supports following features:
  * AB-type polymer melts in bulk
  * Arbitrary acyclic branched polymers
  * Arbitrary mixtures of block copolymers and homopolymers (+ 1 random copolymer)
  * Box size determination by stress calculation (SCFT only)
  * Leimkuhler-Matthews method for updating exchange field (L-FTS only) (**beta**)

This open-source code is distributed under the Apache license 2.0 instead of GPL. This license is one of the permissive software licenses and has minimal restrictions.

# Dependencies
#### Linux System

#### C++ Compiler
  Any C++ compiler that supports C++14 standard or higher. To use MKL, install Intel oneAPI toolkit (without Intel Distribution for Python).

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
conda create -n lfts python=3.9 cmake=3.19 make conda \
      git pybind11=2.9 scipy openmpi pyyaml  
# Activate virtual environment  
conda activate lfts  
# Download L-FTS  
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
conda env remove -n lfts  
```
# User Guide
+ To use this library, first activate virtual environment by typing `conda activate lfts` in command line. In Python script, import the package by adding  `from langevinfts import *`. To learn how to use it, please see 'examples/ComputeConcentration.py'.
+ The SCFT and L-FTS are implemented on the python shared library. Currently, only `AB`-type polymers are supported. To understand the entire process of simulations, please see sample scripts in `examples/scft_single_file` and `examples/fts_single_file`, and use sample scripts in the `examples/scft` and `examples/fts` to perform actual simulations.
  + Set 'reduce_gpu_memory_usage=True' (default: False) if GPU memory space is insufficient to run your simulation. Instead, performance is reduced by 10 ~ 65% depending on chain model and box size. As an example, please see 'examples/scft/BottleBrushLamella3D.py'.
  + To use two GPUs, set `os.environ["LFTS_NUM_GPUS"]="2"`. This is useful if your GPUs do not support high performance in double precision, but only reduces simulation time by 5-40%. Simulation time may increase depending on the number of grids, number of segments, and GPU environment. Check the performance first. As an example, see 'examples/scft/A15.py'.
  + Set 'use_superposition=False, (default: True) if you want to use 'pseudo.get_polymer_concentration()', which returns block-wise concentrations of a selected polymer species, and 'pseudo.get_chain_propagator()', which returns a propagator of a selected branch.
  + If your SCFT calculation does not converge, reduce the "am.mix_min" (default:0.1) and "am.mix_init" (default:0.1) in parameter set. Please see 'examples/scft/BottleBrushLamella3D.py'.
  + The default platform is cuda for 2D and 3D, and cpu-mkl for 1D.
  + Use FTS in 1D and 2D only for the tests. It does not have a physical meaning.
  + To run simulation using only 1 CPU core, set `os.environ["OMP_MAX_ACTIVE_LEVELS"]="0"` in the python script. As an example, please see 'examples/scft/Gyroid.py'.
+ If your ultimate goal is to use deep learning boosted L-FTS, you may use the sample scripts of DL-FTS repository. (https://github.com/yongdd/deep-langevin-fts) (One can easily turn on/off deep learning from the scripts.)
+ The unit of length in this library is *aN^(1/2)* for both `Continuous` and `Discrete` chain models, where *a* is a reference statistical segment length and *N* is a reference polymerization index. The fields acting on chain are defined as `per reference chain` potential instead of `per reference segment` potential. The same notation is used in [*Macromolecules* **2013**, 46, 8037]. If you want to obtain the same fields used in [*Polymers* **2021**, 13, 2437], multiply *ds* to each field. Please refer to [*J. Chem. Phys.* **2014**, 141, 174103] to learn how to formulate polymer mixtures composed of multiple distinct polymers in the reference polymer length unit.

+ Open-source has no warranty. Make sure that this program reproduces the results of previous SCFT and FTS studies, and also produces reasonable results. For acyclic branched polymers adopting the `Continuous` model with an even number of contour steps, the results must be identical to those of PSCF (https://github.com/dmorse/pscfpp) within the machine precision. For AB diblock copolymers adopting the `Discrete` model, the results must be identical to those of code in [*Polymers* **2021**, 13, 2437].
+ It must produce the same results within the machine precision regardless of platform (CUDA or MKL), use of superposition, and use of GPU memory saving option. After changing 'platform' and 'use_superposition', run a few iterations with the same simulation parameters. And check if it outputs the same results.

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

#### Encoding for Chain Propagators
  1. To obtain the statistics of polymers, it is necessary to compute chain propagator of each block for each direction (forward and backward), and thus, there are *2n* propagators for a branched polymer composed of *n* blocks. The initial condition of each propagator is multiplication of the propagators at the junction, or grafting density of the chain end. For the efficient computation, information of each propagator is encoded to a formatted text code that contains information of sub block(s), monomer type, and segment number. From now on, we will refer to it as propagator code. Each code is in the following format: `(B1B2...)KN`, where `K` is monomer type and `N` is the segment number. `B1, B2, ...` are propagator code(s) of sub block(s), which means that the propagator code is recursively defined. `B1, B2, ...` should be enumerated in parenthesis in ascending order based on ASCII code. The parenthesis is omitted if there is no sub block. This representation is unique regardless of permutation of connected branches, so that it can be used to avoid recomputation of the same branch shapes. In the following explanation, all linear polymers, including homopolymers, are considered as a special case of acyclic branched polymers. 
  2. For a linear homopolymer, the propagator code for each direction is concatenation of monomer type and segment number without parenthesis (e.g. `A10`, `B20`, ...). If three linear homopolymers, `B5`, `A5` and `C5`, are attached to one end of a homopolymer `A10`, the propagator code of block `A10` to the direction of the other end is `(A5B5C5)A10`. Note that propagator code for each sub block(s) must be enumerated in ascending order after sorting their propagator codes based on ASCII code, so that it can be uniquely represented. Let us consider a more complex example. If `(A5B6)B6` and `(A3B6)A6` are connected to one end of a `B4` homopolymer, the propagator code to the direction of the other end is `((A3B6)A6(A5B6)B6)B4`. Information of sub blocks are recursively enumerated in ascending order in the parenthesis. This propagator code can be readily obtained by using a recursive function. For exercise, let us find all propagator codes in linear ABC triblock copolymer composed of `A4`, `B4`, and `C4` blocks. We need to find propagator code for each block and each direction, so there are 6 propagator codes.
      + `A4` block and forward direction: There is no connected block at the chain end, so it is just `A4`.
      + `A4` block and backward direction: `A4` block is connected to `B4` block, and `B4` is connected to `C4`, so it becomes `((C4)B4)A4`.
      + `B4` block and forward direction: `B4` block is connected to `A4` block and there is chain end at `A4`, so it is `(A4)B4`.
      + `B4` block and backward direction: `B4` block is connected to `C4` block and there is chain end at `C4`, so it is `(C4)B4`.
      + `C4` block and forward direction: There is no connected block at the chain end, so it is just `C4`.
      + `C4` block and backward direction: `C4` block is connected to `B4` block, and `B4` is connected to `A4`, so it becomes `((A4)B4)C4`.
  3. An arbitrary acyclic branched polymer can be considered as a connected acyclic graph. In this view, blocks and junctions of a polymer are considered as edges and nodes (or vertices) of a graph, respectively, and each edge contains information of segment number and monomer type. For explanation, terminologies for rooted tree, which is a tree that has a root node and commonly used as a data structure in computer science, will be used such as child, parent, leaf, height, subtree, etc.
  4. A key of propagator is defined as a text that rightmost segment number is truncated from a its propagator code, that is `(B1B2...)K`. The height of a propagator can be determined by counting the number of open parenthesis of its key at the leftmost side. For instance, the heights of `A`, `(A4A4)A`, and `((A4)B5)A` are 0, 1, and 2, respectively.
  5. When a new polymer is added, `depth first search` is performed to check whether a given polymer graph contains a cycle or isolated points.

#### Compute Propagators with Minimal Iterations
  1. In order to avoid recomputation of the propagators of the same branch shape and to achieve minimal propagator iterations, a bottom-up `dynamic programming` approach is adopted. In the first step, enumerate all propagator codes for each polymer, for each block, and for each direction. In the second step, find the propagator codes that have the largest `N` among the same keys `(B1B2...)K`, and remove other propagator codes. This is because a propagator of `(B1B2...)KN1` contains a propagator of `(B1B2...)KN2`, if `N1` is equal to or greater than `N2` and the two keys are same. As a result, we can obtain essential propagator codes, and compute the propagators without duplications. Next, we compute the propagators with height of 0. Then, using these solutions, the propagators of branches with height of 1 are calculated. The last step is repeated up to the propagators of the largest height.
      + Example 1: There is a mixture of two `A` homopolymers with segment numbers of 4 and 6, respectively. It is not necessary to separately compute the propagators of shorter `A` homopolymers and complementary propagators. Thus, only necessary computations are from *q*(**r**,`A0`) to *q*(**r**,`A6`). Let us find the only essential propagators using propagator codes. In the first step, enumerate all propagator codes, and we obtain [`A6`, `A6`, `A4`, `A4`] in descending order. In the second step, find only essential propagator codes, and we only have [`A6`]. Now, we have found that we only need to compute the propagators from *q*(**r**,`A0`) to *q*(**r**,`A6`).
      + Example 2: There is 3-arm star `A` homopolymers. Each arm is composed of 4 `A` segments. It is only necessary to compute the propagators of one arm. Let us find the only essential propagators using propagator codes. In the first step, enumerate all propagator codes, and we obtain [`A4`, `A4`, `A4`, `(A4A4)A4`, `(A4A4)A4`, `(A4A4)A4`] in descending order. In the second step, find only essential propagator codes, and we only have [`A4`, `(A4A4)A4`]. Now, we have found that we only need to compute the propagators from *q*(**r**,`A0`) to *q*(**r**,`A4`), and from *q*(**r**,`(A4A4)A0`) to *q*(**r**,`(A4A4)A4`). 
      + Example 3: There is a mixture of `A` homopolymers and symmetric ABA triblock copolymers. The `A` homopolymer is composed of 6 `A` segments. For the ABA triblock, 4 `A`, 5 `B` and 4 `A` segments are linearly connected. Let us find the only essential propagators using propagator codes. In the first step, enumerate all propagator codes, and we obtain [`A6`, `A6`, `A4`, `A4`,`(A4)B5`, `(A4)B5`, `((A4)B5)A4`, `((A4)B5)A4`] in descending order. In the second step, find only essential propagator codes, and we only have [`A6`, `(A4)B5`, `((A4)B5)A4`]. Now, we have found that we only need to compute the propagators from *q*(**r**,`A0`) to *q*(**r**,`A6`), from *q*(**r**,`(A4)B0`) to *q*(**r**,`(A4)B5`), and from *q*(**r**,`(((A4)B5)A0`) to *q*(**r**,`((A4)B5)A4`).
  2. If we want to obtain all propagators separately, above approach is optimal. In the most cases, however, we only want to obtain the full partition functions and concentration of each monomer type. In these cases, the concentrations of side chains can be efficiently computed by exploiting linearity of the modified diffusion equation. Since propagators of side chains appear as summations in the equation for concentration computation, we can use the superposition principle to obtain the sum of propagators. This trick was first introduced in [*Phys. Rev. E* **2002**, 65, 030802(R)] for polymer brushes, and adopted for bottlebrushes in [*Macromolecules* **2019**, 52, 1794]. This idea is generalized in this library for arbitrary acyclic branched polymers (Implementation is complicated. It will take some time to add a detail explanation). Note that when this trick is utilized (e.g., 'use_superposition' is set to 'True'), we cannot trace block-wise concentrations and individual propagators, and thus 'pseudo.get_polymer_concentration()' and 'pseudo.get_chain_propagator()' will be disabled. For continuous chain model, it is not always optimal because of our simpson's rule implementation. The propagators can obtained with minimal iterations for the following cases. 
      + Continuous chain model is adopted, and all segment numbers are even.
      + Continuous chain model is adopted, and all segment numbers are odd.
      + Discrete chain model is adopted.

#### Scheduling for the Parallel Computations
  1. If one propagator is independent of the other propagator, they can be computed in parallel. For instance, the propagator and the complementary propagator of AB diblock copolymers can be computed in parallel. Unfortunately, scheduling propagator computations of a mixture of arbitrary acyclic branched polymers is a complex problem. As far as I understand, scheduling tasks to minimize the makespan is an NP-hard problem.
  2. A naive implementation is to compute propagators of the same height in parallel. For instance, let us consider AB diblock copolymer with 6 `A` segments and 4 `B`. `A` and `B`, whose heights are 0, can be computed in parallel up to `A6` and `B4`, respectively. After computations of height of 0 is done, `(A6)B` and `(B4)A`, whose height is 1, can be computed in parallel up to `(A6)B4` and `(B4)A6`, respectively. However, this plan is not optimal for this asymmetric AB diblock copolymers. Because `(B4)A` has to wait until `A6` is finished, but `(B4)A` can be computed as soon as `B4` is computed. The optimal parallel plan follows. First, compute `A` and `B` up to `A4` and `B4` in parallel. Next, continue the computations of `A` and `(B4)A` up to `A6` and `(B4)A2` in parallel. Finally, compute `(A6)B` and `(B4)A` in parallel until `(A6)B4` and `(B4)A6`.
  3. This library adopts a simple `greedy algorithm`, which is heuristic and is not guaranteed to minimize the makespan, but it is optimal at least for the AB diblock copolymer. Following explanation is based on CPU version, but the same algorithm is utilized for scheduling cuFFT batches in the CUDA version. First, prepare a scheduler for each CPU core. They are initially empty. Second, select keys with height of 0, and insert them one by one into the scheduler whose total reserved computation time is the smallest. Next, select keys with height of 1, find the time that each key is ready by resolving its dependencies, sort them with the resolved time in ascending order, and insert them one by one into the scheduler whose total reserved computation time is the smallest. Repeat this process until all propagators are scheduled. Increase the height by 1 for each iteration. The number of active CPU cores can vary during the computations depending on polymer architectures and the dependencies of branches. 
  4. It is not good idea to assign a CPU core to each scheduler and then run them independently. Because their computations will not be synchronized, it is not guaranteed to satisfy the dependencies of propagators. So, we need to synchronize the computations whenever computations of propagators are finished or computations of new propagators begin. Thus, entire timeline is split to smaller time spans based on the schedule planned above so that the computations can be synchronized for each time span.
  5. The CPU version uses up to 4 CPUs, and the CUDA version uses batched cuFFT with a maximum batch size of 2.

#### Reducing GPU Memory Usage
  1. Propagators of all segments are stored in the GPU's global memory to minimize data transfer between main memory and global memory, because data transfer operations are expensive. However, this method limits the sizes of the grid number and segment number. If the GPU memory space is not enough to run simulations, the propagators should be stored in main memory instead of GPU memory. To reduce data transfer time, `device overlap` can be utilized, which simultaneously transfers data and executes kernels. An example applied to AB diblock copolymers is provided in the supporting information of [*Macromolecules* **2021**, 54, 11304]. To enable this option, set 'reduce_gpu_memory_usage' to 'True' in the example script. If this option is enabled, the factory will create an instance of CudaPseudoReduceMemoryContinuous or CudaPseudoReduceMemoryDiscrete. 
  2. In addition, when 'reduce_gpu_memory_usage' is enabled, field history for Anderson Mixing is also stored in main memory, and the factory will create CudaAndersonMixingReduceMemory.

#### Platforms  
  This program is designed to run on different platforms such as MKL and CUDA, and there is a family of classes for each platform. To produce instances of these classes for given platform, `abstract factory pattern` is adopted.   

#### Anderson Mixing  
  It is necessary to store recent history of fields during iteration. For this purpose, it is natural to use `circular buffer` to reduce the number of array copies.

#### Python Binding  
  `pybind11` is utilized to generate Python interfaces for the C++ classes.  
  https://pybind11.readthedocs.io/en/stable/index.html   

# References
#### Polymer Mixture
+ D. Düchs, K. T. Delaney and G. H. Fredrickson, A multi-species exchange model for fully fluctuating polymer field theory simulations. *J. Chem. Phys.* **2014**, 141, 174103
#### CUDA Implementation
+ G.K. Cheong, A. Chawla, D.C. Morse and K.D. Dorfman, Open-source code for self-consistent field theory calculations of block polymer phase behavior on graphics processing units. *Eur. Phys. J. E* **2020**, 43, 15
+ D. Yong, Y. Kim, S. Jo, D. Y. Ryu, and J. U. Kim, Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces. *Macromolecules* **2021**, 54, 11304
#### Langevin FTS
+ M.W. Matsen, and T.M. Beardsley, Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation, *Polymers* **2021**, 13, 2437   
#### Exchange Field Update Algorithm
+ B.Vorselaars, Efficient Langevin and Monte Carlo sampling algorithms: the case of
field-theoretic simulations, *J. Chem. Phys.* **2023**, 158, 114117

# Citation
Daeseong Yong, and Jaeup U. Kim, Accelerating Langevin Field-theoretic Simulation of Polymers with Deep Learning, *Macromolecules* **2022**, 55, 6505  