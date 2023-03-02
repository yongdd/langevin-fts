# Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) for Python

# Features
* SCFT and L-FTS
* Arbitrary Acyclic Complex Branched Polymers (**beta**)
* Arbitrary Mixtures of Block Copolymers and Homopolymers (+ 1 Random Copolymer) (**beta**)
* AB-Type Polymer Melts (**The Python library itself can compute concentrations of polymers composed of any number of monomer types**)
* Automatic Optimization to Compute Polymer Concentration with Minimal Propagator Iterations (**beta**)
* Conformational Asymmetry
* Box Size Determination by Stress Calculation (SCFT only)
* Chain Model: Continuous, Discrete
* Periodic Boundaries
* 3D, 2D and 1D
* Pseudospectral Method, Anderson Mixing
* Platforms: MKL (CPU) and CUDA (GPU)

# Dependencies
#### Linux System

#### C++ Compiler
  Any C++ compiler that supports C++14 standard or higher. To use MKL, install Intel oneAPI toolkit (without Intel Distribution for Python).

#### CUDA Toolkit
  https://developer.nvidia.com/cuda-toolkit   
  Required for the GPU computation. If it is not installed, ask admin for its installation.

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
+ This is not an application but a library for polymer field theory simulations, and you need to write your own program using Python language. It requires a little programming, but this approach provides flexibility and you can easily customize your applications. To use this library, first activate virtual environment by typing `conda activate lfts` in command line. In Python script, import the package by adding  `from langevinfts import *`. This library itself can calculate the partition functions and concentrations of any mixtures of any acyclic branched polymers composed of multiple monomer types. To learn how to use it, please see `examples/ComputeConcentration.py`. 
+ The SCFT and L-FTS are implemented on the top of this Python library as Python scripts. Currently, only `AB`-type polymers are supported. To understand the entire process of simulations, please see sample scripts in `examples/scft_single_file` and `examples/fts_single_file`, and use sample scripts in the `examples/scft` and `examples/fts` to perform actual simulations.
  + Set 'reduce_gpu_memory_usage' to 'True' (Default: False) if GPU memory space is insufficient to run your simulation. Instead, performance is reduced by 5 ~ 60% depending on chain model and box size. As an example, please see 'examples/scft/BottleBrushLamella3D.py'.
  + Set 'use_superposition' to 'False, (Default: True) if you want to use 'pseudo.get_polymer_concentration()', which returns block-wise concentrations of a selected polymer species, and 'pseudo.get_partial_partition()', which returns a partial partition function of a selected branch.
  + The default platform is cuda for 2D and 3D, and cpu-mkl for 1D.
  + Use FTS in 1D and 2D only for the tests. It does not have a physical meaning.
  + To run simulation using only 1 CPU core, set `os.environ["OMP_MAX_ACTIVE_LEVELS"]="0"` in the python script. As an example, please see 'examples/scft/Gyroid.py'.
+ If your ultimate goal is to use deep learning boosted L-FTS, you may use the sample scripts of DL-FTS repository. (One can easily turn on/off deep learning from the scripts.)
+ The unit of length in this library is *aN^(1/2)* for both `Continuous` and `Discrete` chain models, where *a* is a reference statistical segment length and *N* is a reference polymerization index. The fields acting on chain are defined as `per reference chain` potential instead of `per reference segment` potential. The same notation is used in [*Macromolecules* **2013**, 46, 8037]. If you want to obtain the same fields used in [*Polymers* **2021**, 13, 2437], multiply *ds* to each field. Please refer to [*J. Chem. Phys.* **2014**, 141, 174103]  to learn how to formulate polymer mixtures composed of multiple distinct polymers in the reference polymer length unit.

+ Open-source has no warranty. Make sure that this program reproduces the results of previous SCFT and FTS studies, and also produces reasonable results. For acyclic branched polymers adopting the `Continuous` model with an even number of contour steps, the results must be identical to those of PSCF (https://github.com/dmorse/pscfpp) within the machine precision. For AB diblock copolymers adopting the `Discrete` model, the results must be identical to those of code in [*Polymers* **2021**, 13, 2437].
+ It must produce the same results within the machine precision regardless of platform (CUDA or MKL) and use of superposition. After changing 'platform' and 'use_superposition', run a few iterations with the same simulation parameters. And check if it outputs the same results.

+ Matlab and Python tools for visualization and renormalization are included in `tools` folder.

# Contribution
+ Most of python scripts implemented with this library are welcome. They could be very simple scripts for specific polymer morphologies, or modified versions of `scft.py`, `lfts.py`, etc.
+ Changing C++/CUDA codes by yourself is not recommended. If you want to add new features to this python library, please send me sample codes.
+ They should contain sample results, test codes, or desired outputs to check whether they work correctly.
+ There should be relevant published literatures.
+ Currently, this library is updated without considering its compatibility with previous versions. I will keep managing the contributed codes so that they can be executed in the updated version.
+ **Contributed codes must not contains GPL-licensed codes.**
+ Please do not send me exclusive codes of your lab. Make sure that they are allowed as open-source.
+ Any suggestions and advices are welcome, but they could not be reflected.

# Developer Guide
#### Platforms  
  This program is designed to run on different platforms such as MKL and CUDA, and there is a family of classes for each platform. To produce instances of these classes for given platform, `abstract factory pattern` is adopted.   

#### Efficient Computation of Partial Partition Functions and Polymer Concentrations
  1. To obtain the statistics of polymers, it is necessary to compute partial partition function of each block for each direction (forward and backward), and thus, there are *2n* partial partition functions for a branched polymer composed of *n* blocks. The initial condition of each partial partition function is multiplication of the partial partition function of blocks that are connected to the junction, or grafting density for the chain end. For the efficient computation, each partial partition function has a text code that contains information of connected blocks. This should be unique regardless of permutation of connected blocks or branches, so that it can be used to avoid recomputation of the same branch shapes. In the following context, all linear polymers, including homopolymers, are considered as a special case of acyclic branched polymers. 
  2. An arbitrary acyclic branched polymer can be considered as a connected acyclic graph. In this view, blocks and junctions of a polymer are considered as edges and nodes (or vertices) of a graph, respectively, and each edge contains information of segment number and monomer type. For explanation, terminologies for rooted tree will be used such as child, parent, leaf, height, subtree, etc. 
  3. For a linear homopolymer, which can be considered as the most simplest branched polymer, the text code for each direction is concatenation of monomer type and segment number (e.g. `A10`, `B20`, ...). For an arbitrary acyclic branched polymer, the text code of a partial partition function starting at the source node `v` and ending at the target node `u` can be described by using a rooted tree, which is a tree that has a root node and commonly used as a data structure in computer science. The node `u` is selected as the root node, and other edges connected to node `u` will be ignored, and thus the root node has only one child node `v`. If there are edges from node `v` to the connected subtrees, text code of partition function of each edge is enumerated in parenthesis in ascending order and it is concatenated with monomer type and segment number of the edge of `v → u`. For instance, if three linear homopolymers, `B5`, `A5` and `C5`, are attached to one end of a homopolymer `A10`, the text code of the partial partition function for `A10` to the direction of the other end is `(A5B5C5)A10`. Note that text code of partial partition function for each connected edge must be enumerated in ascending order after sorting their text codes, so that it can be uniquely represented. Let us consider a more complex example. If `(A5B6)B6` and `(A3B6)A6` are connected to one end of a `B4` homopolymer, the text code to the direction of the other end is `((A3B6)A6(A5B6)B6)B4`. Information of subtrees are recursively enumerated in ascending order in the parenthesis. This text code can be readily obtained by using a recursive function. For exercise, let us find all text codes in linear ABC triblock copolymer composed of `A4`, `B4`, and `C4` blocks. We need to find text codes for each block and each direction, so there are 6 text codes.
      + `A4` block and forward direction: There is no connected block at the chain end, so it is just `A4`.
      + `A4` block and backward direction: `A4` block is connected to `B4` block, and `B4` is connected to `C4`, so it becomes `((C4)B4)A4`.
      + `B4` block and forward direction: `B4` block is connected to `A4` block and there is chain end at `A4`, so it is `(A4)B4`.
      + `B4` block and backward direction: `B4` block is connected to `C4` block and there is chain end at `C4`, so it is `(C4)B4`.
      + `C4` block and forward direction: There is no connected block at the chain end, so it is just `C4`.
      + `C4` block and backward direction: `C4` block is connected to `B4` block, and `B4` is connected to `A4`, so it becomes `((A4)B4)C4`.

  4. Each rooted tree is considered as a branch. A key of branch is defined as a text that rightmost segment number is truncated from a its text code. The height of a branch can be determined by counting the number of open parenthesis of its key. For instance, the heights of `A`, `(A4A4)A`, and `((A4)B5)A` are 0, 1, and 2, respectively.
  5. In order to avoid recomputation of the partial partition functions of the same branch shape and to achieve minimal propagator iterations, a bottom-up `dynamic programming` approach is adopted. The partial partition functions of branches with height of 0 are first computed. Then, using these solutions, the partial partition functions of branches with height of 1 are calculated. The last step is repeated up to the most tallest branches.
      + Example 1: There is a mixture of two `A` homopolymers with segment numbers of 4 and 6, respectively. It is not necessary to separately compute the partial partition functions of shorter `A` homopolymers and complementary partition functions. Thus, only necessary computations are from *q*(**r**,`A0`) to *q*(**r**,`A6`). Let us find the only essential computations using text codes. In the first step, enumerate all text codes for each polymer and for each direction, and we obtain [`A6`, `A6`, `A4`, `A4`] in descending order. In the second step, remove duplicated codes and leave only text codes have the largest segment number (which refers to the rightmost segment number and do not care about branches in the parentheses), and we only have [`A6`]. Now, we have found that we only need to compute the partial partition functions from *q*(**r**,`A0`) to *q*(**r**,`A6`).
      + Example 2: There is 3-arm star `A` homopolymers. Each arm is composed of 4 `A` segments. It is only necessary to compute the partial partition functions of one arm. Let us find the only essential computations using text codes. In the first step, enumerate all text codes for each block and for each direction, and we obtain [`A4`, `A4`, `A4`, `(A4A4)A4`, `(A4A4)A4`, `(A4A4)A4`] in descending order. In the second step, remove duplicated codes and leave only text codes have the largest segment number, and we only have [`A4`, `(A4A4)A4`]. Now, we have found that we only need to compute the partial partition functions from *q*(**r**,`A0`) to *q*(**r**,`A4`), and from *q*(**r**,`(A4A4)A0`) to *q*(**r**,`(A4A4)A4`). 
      + Example 3: There is a mixture of `A` homopolymers and symmetric ABA triblock copolymers. The `A` homopolymer is composed of 6 `A` segments. For the ABA triblock, 4 `A`, 5 `B` and 4 `A` segments are linearly connected. Let us find the only essential computations using text codes. In the first step, enumerate all text codes for each block and for each direction, and we obtain [`A6`, `A6`, `A4`, `A4`,`(A4)B5`, `(A4)B5`, `((A4)B5)A4`, `((A4)B5)A4`] in descending order. In the second step, remove duplicated codes and leave only text codes have the largest segment number, and we only have [`A6`, `(A4)B5`, `((A4)B5)A4`]. Now, we have found that we only need to compute the partial partition functions from *q*(**r**,`A0`) to *q*(**r**,`A6`), from *q*(**r**,`(A4)B0`) to *q*(**r**,`(A4)B5`), and from *q*(**r**,`(((A4)B5)A0`) to *q*(**r**,`((A4)B5)A4`).
  6. If we want to obtain all partial partition functions, above approach is optimal. In the most cases, however, we want to obtain only full partition functions and polymer concentrations. In these cases, the polymer concentrations of side chains can be efficiently computed by exploiting the superposition principle, since the modified diffusion equation is a linear equation. This trick was first introduced in [*Phys. Rev. E* **2002**, 65, 030802(R)] for polymer brushes, and adopted for bottlebrushes in [*Macromolecules* **2019**, 52, 1794] (see supporting information). This idea is generalized in this library even for various side chains with different lengths (implementation detail will be added). Note that when this trick is utilized, we cannot trace block-wise concentrations and individual partial partition functions. Thus, 'pseudo.get_polymer_concentration()' and 'pseudo.get_partial_partition()' become disabled, if 'use_superposition' is set to 'True'. The polymer concentrations are obtained with minimal propagator iterations for the following cases. In other cases, it is at least sub optimal.
      + Continuous chain model is adopted, and all segment numbers are even.
      + Continuous chain model is adopted, and all segment numbers are odd.
      + Discrete chain model is adopted.
  7. When a new polymer is added, `depth first search` is performed to check whether a given polymer graph contains a cycle or isolated points.

#### Scheduling for the Parallel Computations
  1. If one partial partition function is independent of the other partial partition function, they can be computed in parallel. For instance, the partial partition function and the complementary partial partition function of AB diblock copolymers can be computed in parallel. Unfortunately, scheduling partial partition function computations of a mixture of arbitrary acyclic branched polymers is a complex problem. As far as I understand, scheduling tasks to minimize the makespan for tasks of arbitrary time lengths is an NP-hard problem. 
  2. A naive implementation is to compute partial partition functions of the same height in parallel. For instance, let us consider AB diblock copolymer with 6 `A` segments and 4 `B`. `A` and `B`, whose heights are 0, can be computed in parallel up to `A6` and `B4`, respectively. After computations of height of 0 is done, `(A6)B` and `(B4)A`, whose height is 1, can be computed in parallel up to `(A6)B4` and `(B4)A6`, respectively. However, this plan is not optimal for this asymmetric AB diblock copolymers. Because `(B4)A` has to wait until `A6` is finished, but `(B4)A` can be computed as soon as `B4` is computed. The optimal parallel plan follows. First, compute `A` and `B` up to `A4` and `B4` in parallel. Next, continue the computations of `A` and `(B4)A` up to `A6` and `(B4)A2` in parallel. Finally, compute `(A6)B` and `(B4)A` in parallel until `(A6)B4` and `(B4)A6`.
  3. This library adopts a simple `greedy algorithm`, which is heuristic and is not guaranteed to minimize the makespan, but it is optimal at least for the AB diblock copolymer. Following explanation is based on CPU version, but the same algorithm is utilized for scheduling cuFFT batches in the CUDA version. First, prepare a scheduler for each CPU core. They are initially empty. Second, select keys with height of 0, and insert them one by one into the scheduler whose total reserved computation time is the smallest. Next, select keys with height of 1, find the time that each key is ready by resolving its dependencies, sort them with the resolved time in ascending order, and insert them one by one into the scheduler whose total reserved computation time is the smallest. Repeat this process until all keys are scheduled. Increase the height by 1 for each iteration. The number of active CPU cores can vary during the computations depending on polymer architectures and the dependencies of branches. 
  4. It is really bad idea to assign a CPU core to each scheduler and then run them independently. Because their computations will be not synchronized, it is not guaranteed to satisfy the dependencies of branches. So, we need to synchronize the computations whenever computations of branches are finished or computations of new branches begin. Thus, entire timeline is split to smaller time spans based on the schedule planned above so that the computations can be synchronized for each time span.
  5. The CPU version uses up to 4 CPUs, and the CUDA version uses batched cuFFT with a maximum batch size of 2.

#### Reducing GPU Memory Usage
  Partial partition functions of all segments are stored in the GPU's global memory to minimize data transfer between main memory and global memory, because data transfer operations are expensive. However, this method limits the sizes of the grid number and segment number. If the GPU memory space is not enough to run simulations, the partial partition functions should be stored in main memory instead of GPU memory. To reduce data transfer time, `concurrent data copy and kernel execution` can be utilized, which simultaneously transfers data and executes kernels. An example applied to AB diblock copolymers is provided in the supporting information of [*Macromolecules* **2021**, 54, 11304]. In the current implementation, the polymer concentrations are computed on the CPU, whereas all calculations are parallelized on the GPU in the previous implementation. To enable this GPU memory saving option, set 'reduce_gpu_memory_usage' to 'True' in the example script. If this option is enabled, the factory will create an instance of CudaPseudoReduceMemoryContinuous or CudaPseudoReduceMemoryDiscrete.

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

# Citation
Daeseong Yong, and Jaeup U. Kim, Accelerating Langevin Field-theoretic Simulation of Polymers with Deep Learning, *Macromolecules* **2022**, 55, 6505  