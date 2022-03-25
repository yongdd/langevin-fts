# Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) for Python

# Features
* AB Diblock Copolymer Melt
* Periodic Boundaries  
* 3D, 2D and 1D (Caution! FTS in 1D and 2D has no physical meaning)
* Pseudospectral Implmentation using MKL, FFTW and CUDA
* Gaussian and Discrete Chain Models
* Linux

# Dependencies
#### C++ Compiler
  Any C++ compiler that supports C++14 standard or higher. If you want to use MKL, install Intel compiler.

#### CUDA Toolkit 
  https://developer.nvidia.com/cuda-toolkit  

#### Anaconda
  https://www.anaconda.com/

* * *
Environment variables must be set so that `nvcc` and `conda` can be executed in the command line (Type `which nvcc` and `which conda` to check the installation).
  
# Compiling
  `conda create -n envlfts python=3.8 cmake=3.19 conda swig scipy fftw openmpi`  
  `conda activate envlfts`  
  `git clone https://github.com/yongdd/langevin-fts.git`  
  `cd langevin-fts`  
  `mkdir build`  
  `cd build`  
  `cmake ../`  
  `make`   
  `make test`   
  `make install`
  
* * *
  You can specify your C++ compiler with following command.   
  `cmake ../  -DCMAKE_CXX_COMPILER=[Your CXX Compiler, e.g. "icpc", "g++"]`  

# User Guide
+ This is not an application but a library for the field-based simulation, and you need to write your own program using Python language. It requires a little programming, but this approach provides flexibility and you can easily customize your applications. Basic SCFT and L-FTS implementations are provided in  `examples` folder.
+ To use this library, first activate virtual environment by typing `conda activate envlfts` in command line. In python, import the package by adding  `from langevinfts import *`.
+ Be aware that unit of length in this program is end-to-end chain length *aN^(1/2)*, not gyration of radius *a(N/6)^(1/2)*, where *a* is statistical segment length and *N* is polymerziation index.  
+ Open source has no warranty. Make sure that this program reproduces the results of previous FTS studies, and also produces resonable results.  

# Developer Guide
#### Platforms  
  This program is designed to run on different platforms such as MKL, FFTW and CUDA, and there is a family of classes for each platform. To produce instances of these classes for given platform `abstract factory pattern` is adopted.

#### Anderson Mixing  
  It is neccesery to store recent history of fields during iteration. For this purpose, it is natural to use `circular buffer` to reduce the number of array copys. If you do not want to use such data structure, please follow the code in [*Polymers* **2021**, 13, 2437]. The performance loss is only marginal.

#### Parameter Parser  
  A parser is implemented using `regular expression` and `deterministic finite automaton` to read input parameters from a file. If you want to modify or improve syntax for parameter file, reimplement the parser using standard tools such as `bison` and `flex`. In python scripts, you can use a `yaml` or `json` file as an input parameter file instead. Using `argparse` is also good option.
  
# References
#### CUDA Implementation
+ G.K. Cheong, A. Chawla, D.C. Morse and K.D. Dorfman, Open-source code for self-consistent field theory calculations of block polymer phase behavior on graphics processing units. *Eur. Phys. J. E* **2020**, 43, 15
#### Gaussian Chain Model
+ T.M. Beardsley, R.K.W. Spencer, and M.W. Matsen, Computationally Efficient Field-Theoretic Simulations for Block Copolymer Melts, *Macromolecules* **2019**, 52, 8840   
+ M.W. Masen, Field theoretic approach for block polymer melts: SCFT and FTS, *J. Chem. Phys.* **2020**, 152, 110901   
#### Discrete Chain Model
+ T.M. Beardsley, and M.W. Matsen, Fluctuation correction for the order–disorder transition of diblock copolymer melts, *J. Chem. Phys.* **2021**, 154, 124902   
+ M.W. Matsen, and T.M. Beardsley, Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation, *Polymers* **2021**, 13, 2437   
####  Field-Update Algorithms
+ D.L. Vigil, K.T. Delaney, and G.H. Fredrickson, Quantitative Comparison of Field-Update Algorithms for Polymer SCFT and FTS, *Macromolecules* **2021**, 54, 21, 9804
#### My Work
+ D. Yong, Y. Kim, S. Jo, D.Y. Ryu, and .U. Kim*, Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces, *Macromolecules* **2021**, 54, 11304   
