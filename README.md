# Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) for Python

# Features
* AB Diblock Copolymer Melt
* Periodic Boundaries  
* 3D, 2D and 1D (Caution! FTS in 1D and 2D has no physical meaning)
* Pseudospectral Implmentation using MKL, FFTW and CUDA
* Gaussian and Discrete Chain Models

# Dependencies
#### C++ Compiler
  Any C++ compiler that supports C++14 standard or higher, but I recommend to use Intel compilers. They are free and faster than GCC even on AMD CPU. If you want to Intel compilers, install Intel oneAPI Base & HPC Toolkit. 

#### CUDA  
  https://developer.nvidia.com/cuda-toolkit  

#### Anaconda 3.x
  Anaconda is a distribution of the Python pogramming languages for scientific computing.  
  https://www.anaconda.com/

* * *
I tested this program under following environments.  
+ Intel oneAPI Base & Toolkit 2021.3.0  
+ CUDA Toolkit 11.2  
+ OpenMP bundled with Intel Compilers 2021.3.0  

# Compile
  `conda create -n envlfts python=3.8 conda`  
  `conda activate envlfts`  
  `conda install cmake=3.19 swig scipy mkl fftw openmpi mpi4py`   
  `git clone https://github.com/yongdd/langevin-fts.git`  
  `cd langevin-fts`  
  `mkdir build`  
  `cd build`  
  `cmake ../`  
  `make`   
  `make test`   
  `make install`
  
* * *
  You can specify your building flags with following command.   
  `cmake ../  -DCMAKE_CXX_COMPILER=[Your CXX Compiler, e.g. "icpc", "g++"] \`   
  `-DUSE_OPENMP=yes`
  
* * *
  To use this library, first activate virtual environment by typing `conda activate envlfts` in command line.
  In python, import the package by adding  `from langevinfts import *`.
  
# User Guide
+ This is not an application but a library for field-based simulation, and you need to write your own problem using Python language. It requires a little programming, but this approach provides flexibility and you can easily customize your applications. Please look around `examples` folder to understand how to use this library.
+ Be aware that unit of length in this program is end-to-end chain length *aN^(1/2)*, not gyration of radius *a(N/6)^(1/2)*, where *a* is statistical segment length and *N* is polymerziation index.  
+ Open source has no warranty. Make sure that this program reproduces the results of previous FTS studies, and also produces resonable results.  

# Developer Guide
#### Abstract Factory   
  This program is designed to run on different platforms such as FFTW, MKL and CUDA, there is a family of classes for each platform. To produce instances of these classes for given platform `abstract factory pattern` is adopted.

#### Anderson Mixing   
  It is neccesery to store recent history of fields during iteration. For this purpose, it is natural to use `circular buffer` to reduce the number of array copys. If you do not want to use such data structure, please follow the code in [*Polymers* **2021**, 13, 2437]. The performance loss is only marginal.

#### Parser (class ParamParser)   
  A parser is implemented using `regular expression` and `deterministic finite automaton` to read input parameters from a file. If you want to modify or improve syntax for parameter file, reimplement the parser using standard tools such as `bison` and `flex`. Instead, you can use a `yaml` or `json` file as an input parameter file in python scripts. Using `argparse` is also good option.
  
# References
#### Gaussian Chain Model
+ T.M. Beardsley, R.K.W. Spencer, and M.W. Matsen, Computationally Efficient Field-Theoretic Simulations for Block Copolymer Melts, *Macromolecules* **2019**, 52, 8840   
+ M.W. Masen, Field theoretic approach for block polymer melts: SCFT and FTS, *J. Chem. Phys.* **2020**, 152, 110901   
#### Discrete Chain Model
+ T.M. Beardsley, and M.W. Matsen, Fluctuation correction for the order–disorder transition of diblock copolymer melts, *J. Chem. Phys.* **2021**, 154, 124902   
+ M.W. Matsen, and T.M. Beardsley, Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation, *Polymers* **2021**, 13, 2437   
####  Field-Update Algorithms
+ D.L. Vigil, K.T. Delaney, and G.H. Fredrickson, Quantitative Comparison of Field-Update Algorithms for Polymer SCFT and FTS, *Macromolecules* **2021**, 54, 21, 9804
