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
cmake ../   
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
+ The SCFT and L-FTS are implemented on the top of this Python library as Python scripts. Currently, only `AB`-type polymers are supported. To understand the entire process of simulations, please see sample scripts in `examples/scft_single_file` and `examples/fts_single_file`, and use sample scripts in the `examples/scft` and `examples/fts` to perform actual simulations. If your ultimate goal is to use deep learning boosted L-FTS, you may use the sample scripts of DL-FTS repository. (One can easily turn on/off deep learning from the scripts.)
+ The unit of length in this library is *aN^(1/2)* for both `Continuous` and `Discrete` chain models, where *a* is a reference statistical segment length and *N* is a reference polymerization index. The fields acting on chain are defined as `per reference chain` potential instead of `per reference segment` potential. The same notation is used in [*Macromolecules* **2013**, 46, 8037]. If you want to obtain the same fields used in [*Polymers* **2021**, 13, 2437], multiply *ds* to each field. Please refer to [*J. Chem. Phys.* **2014**, 141, 174103]  to learn how to formulate polymer mixtures composed of multiple distinct polymers in the reference polymer length unit.
+ To run simulation using only 1 cpu, set `os.environ["OMP_MAX_ACTIVE_LEVELS"]="0"` in the python script. Please see `examples/scft/Gyroid.py`.
+ Open-source has no warranty. Make sure that this program reproduces the results of previous SCFT and FTS studies, and also produces reasonable results. For acyclic branched polymers adopting the `Continuous` model with an even number of contour steps, the results must be identical to those of PSCF (https://github.com/dmorse/pscfpp) within the machine precision. For AB diblock copolymers adopting the `Discrete` model, the results should be equivalent to those of code in [*Polymers* **2021**, 13, 2437].
+ It must produce the same results within the machine precision regardless of platform (CUDA or MKL) and use of superposition. After changing "platform" and "use_superposition", run a few iterations with the same simulation parameters. And check if it outputs the same results.
+ Use FTS in 1D and 2D only for the test. It does not have a physical meaning.
+ Matlab and Python tools for visualization and renormalization are included in `tools` folder.

# Developer Guide
#### Platforms  
  This program is designed to run on different platforms such as MKL and CUDA, and there is a family of classes for each platform. To produce instances of these classes for given platform, `abstract factory pattern` is adopted.   

#### Efficient Computation of Partial Partition Functions and Concentrations
  1. In the Python library, all polymers including linear polymers are considered as branched polymers.
  2. Whenever you add a new polymer, `depth first search` is performed to check whether a given polymer graph contains a cycle or isolated points.
  3. A polymer block and branches attached thereto are uniquely represented by a text code. The text codes of the attached branches are enumerated in a parenthesis in ascending order.
      + Example 1: There are `A` homopolymers with segment numbers of 6. Suppose the two chain ends are at `s=0` and `s=6`. In forward direction computation of partial partition function, the computed chain length will increase from 0 to 6 (continuous chain), and the text code will change from `A0` to `A6`. In backward direction computation, the text code also will change from `A0` to `A6`.
      + Example 2: Two branches, `B6` and `A3`, are attached to the end of a `A6` homopolymer. At the other end of `A6` homopolymer, the entire chain is expressed as `(A3B6)A6`. Branches should be enumerated in ascending order in the parenthesis.
      + Example 3: Two branches, `(A5B6)B6` and `(A3B6)A6`, are attached to the end of a `B4` homopolymer. At the other end of `B4` homopolymer, the entire chain is expressed as `((A3B6)A6(A5B6)B6)B4`. Branches are also enumerated in ascending order in the parenthesis.
      + Example 4: There are AB diblock copolymers with 4`A` segments and 6`B` segments. In forward direction computation, text codes are `A0`, `(A4)B6`, `A4` at chain ends and the junction. In back direction computation, text codes are `B0`, `(B6)A4`, `B6` at chain ends and the junction.
  4. In order to avoid recomputation of the partial partition functions of the same branch shape and to achieve minimal propagator iterations, a bottom-up `dynamic programming` approach is adopted. The partial partition functions of simplest branches are first computed. Then, using these solutions, the partial partition functions of more complex branches are calculated. The last step is repeated up to the most complex branches.
      + Example 1: There is a mixture of short and long `A` homopolymers with segment numbers of 4 and 6, respectively. It is not necessary to separately compute the partial partition functions of shorter `A` homopolymers and complementary partition functions. Thus, only necessary computations are from *q*(**r**,`A0`) to *q*(**r**,`A6`). In the first step, enumerate all text codes for each polymer and for each direction, then we obtain [`A6`, `A6`, `A4`, `A4`] in descending order. In the second step, remove duplicated codes and leave only codes of the longest ones, then we only have [`A6`]. Now, we know that we only need to compute the partial partition functions from *q*(**r**,`A0`) to *q*(**r**,`A6`).
      + Example 2: There are 3-arm star `A` homopolymers. Each arm is composed of 4 `A` segments. It is only necessary to compute the partial partition functions of one arm. In the first step, enumerate all text codes for each block and for each direction, then we obtain [`A4`, `A4`, `A4`, `(A4A4)A4`, `(A4A4)A4`, `(A4A4)A4`] in descending order.  In the second step, remove duplicated codes and leave only codes of the longest ones, then we only have [`A4`, `(A4A4)A4`]. Now, we know that we only need to compute the partial partition functions from *q*(**r**,`A0`) to *q*(**r**,`A4`), and from *q*(**r**,`(A4A4)A0`) to *q*(**r**,`(A4A4)A4`). In actual implementation, `A` and `(A4A4)A` are used as keys, and 4 and 4 are their corresponding values, respectively.
      + Example 3: There is a mixture of `A` homopolymers and symmetric ABA triblock copolymers. The `A` homopolymer is composed of 6 `A` segments. For the ABA triblock, 4`A`, 5`B` and 4`A` segments are linearly attached. In the first step, enumerate all text codes for each block and for each direction, then we obtain [`A6`, `A6`, `A4`, `A4`,`(A4)B5`, `(A4)B5`, `((A4)B5)A4`, `((A4)B5)A4`] in descending order.  In the second step, remove duplicated codes and leave only codes of the longest one, then we only have [`A6`, `(A4)B5`, `((A4)B5)A4`]. Now, we know that we only need to compute the partial partition functions from *q*(**r**,`A0`) to *q*(**r**,`A6`), from *q*(**r**,`(A4)B0`) to *q*(**r**,`(A4)B5`), and from *q*(**r**,`(((A4)B5)A0`) to *q*(**r**,`((A4)B5)A4`).
  5. When 'use_superposition' is set to 'True', the polymer concentrations of side chains are efficiently computed by exploiting the superposition principle, since the modified diffusion equation is a linear equation. This trick was first introduced in [*Macromolecules* **2019**, 52, 1794] for bottlebrushes (see appendix) and it is generalized in this library (implementation detail will be added). The polymer concentrations are obtained with minimal propagator iterations for the following cases. In other cases, it is at least sub optimal.
      + Continuous chain model is adopted, and all segment numbers are even.
      + Continuous chain model is adopted, and all segment numbers are odd.
      + Discrete chain model is adopted.

#### Scheduling for the Parallel Computations
  1. If one partial partition function is independent of the other partial partition function, they can be computed in parallel. For instance, the partial partition function and the complementary partial partition function of AB diblock copolymers can be computed in parallel. Unfortunately, scheduling partial partition function computations of a mixture of arbitrary acyclic branched polymers is a complex problem.
  2. The height of a branch can be determined by counting the number of open parenthesis of its key. For instance, the heights of `A`, `(A4A4)A`, and `((A4)B5)A` are 0, 1, and 2, respectively. A naive implementation is to compute partial partition functions of the same height in parallel. For instance, let us consider AB diblock copolymer with 6`A` segments and 4`B`. `A` and `B`, whose heights are 0, can be computed in parallel up to `A6` and `B4`, respectively. After computations of height of 0 is done, `(A6)B` and `(B4)A` whose heights are 1 can be computed in parallel up to `(A6)B4` and `(B4)A6`, respectively. However, the optimal parallel schedule follows. First, compute `A` and `B` up to `A4` and `B4` in parallel. Next, continue the computations of `A` and `(B4)A` up to `A6` and `(B4)A2` in parallel. Finally, compute `(A6)B` and `(B4)A` in parallel until `(A6)B4` and `(B4)A6`.
  3. For parallel computation scheduling, this library uses a simple `greedy algorithm`. First, select keys with height of 0, and put them one by one into a stream (one of the CPUs for CPU version or one of the cuFFT batches for CUDA version) scheduler whose reserved total computation time is the smallest. Next, select keys with height of 1, find the time that each key is ready by resolving its dependencies, sort them with the resolved time in ascending order, and put them one by one into a stream scheduler whose reserved total computation time is the smallest. Repeat this process until all keys are scheduled. Increase the height by 1 for each iteration. This approach is heuristic and is not guaranteed to minimize the makespan. As far as I know, scheduling tasks to minimize the makespan is an NP-hard problem.
  4. Whenever computations are completed or new computations begin, it is necessary to reassign jobs to streams. Thus entire timeline is split to smaller time spans based on the schedule planned above. As a result, the number of active streams can be varying during the computations based on the dependencies of branches. For the AB diblock copolymer, always two streams are active.
  5. The CPU version uses up to 4 CPUs, and the CUDA version uses batched cuFFT with a maximum batch size of 2.

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
#### Langevin FTS
+ M.W. Matsen, and T.M. Beardsley, Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation, *Polymers* **2021**, 13, 2437   

# Citation
Daeseong Yong, and Jaeup U. Kim, Accelerating Langevin Field-theoretic Simulation of Polymers with Deep Learning, *Macromolecules* **2022**, 55, 6505  