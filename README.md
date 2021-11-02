# Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS)

# Features
* Diblock Copolymer Melt
* Periodic Boundaries  
* 3D, 2D and 1D (1D and 2D are only for test purpose)
* Pseudospectral Implmentation using MKL, FFTW and CUDA
* Accelerating L-FTS using Deep Learning

# Dependencies
#### 1. C++ Compiler
  Any C++ compiler that supports C++11 standard or higher, but I recommend to use Intel compilers. Install Intel oneAPI Base & HPC Toolkit. They are free and faster than GCC even on AMD CPU.

#### 2. FFT Library
  The modified diffusion equations are solved by pseudospectral method, and that requires a fast Fourirer transform (FFT) library. You can choose from following FFT libraries.

+ **(optional) MKL**   
  Math kernel library (MKL) is bundled with Intel Compilers.  

+ **(optional) FFTW**   
  https://www.fftw.org/
  
+ **CUDA**  
  https://developer.nvidia.com/cuda-toolkit  
  
#### 3. (optional) OpenMP
  Two partial partition functions are calculated simultaneously using open multi-processing (OpenMP) in the CPU implemenation.  

#### 4. CMake 3.17+

#### 5. SWIG
  A tool that connects libraries written in C++ with Python    
  http://www.swig.org/

#### 6. Anaconda 3.x
  Anaconda is a distribution of the Python pogramming languages for scientific computing.  
  https://www.anaconda.com/

* * *
I tested this program under following environments.  
+ C++ Compilers
  + Intel oneAPI Base & Toolkit 2021.3.0   
  + The GNU Compiler Collection 7.5 
+ CUDA Toolkit 11.2
+ OpenMP bundled with Intel Compilers 2021.3.0

# Compile
  `git clone https://github.com/yongdd/Langevin_FTS_Public.git`  
  `cd Langevin_FTS_Public`  
  `mkdir build`  
  `cd build`  
  `cmake ../`  
  `make`   
  `make test`   
  `make install`
* * *
  You can specify your building flags with following command.   
  `cmake ../  -DCMAKE_CXX_COMPILER=[Your CXX Compiler, e.g. "icpc", "g++"] \`   
  `-DCMAKE_INCLUDE_PATH=[Your FFTW Path]/include \`  
  `-DCMAKE_FRAMEWORK_PATH=[Your FFTW Path]/lib \`  
  `-DUSE_OPENMP=yes`
* * *
  In python, import the package by adding  `from langevinfts import *`.
# User Guide

# Developer Guide
  A few things you need to know.  

+ **Object Oriented Programming (OOP)**  
    Basic concepts of OOP such as class, inheritance and dynamic binding.  
    In addtion, some design patterns. (class ParamParser, CudaCommon, AbstractFactory)
+ **CUDA Programming** (./platforms/cuda)  
    This is a introductory book written by NVIDIA members  
  https://developer.nvidia.com/cuda-example  
    Optimizing Parallel Reduction in CUDA  
  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
+ **(optional) Parser** (class ParamParser)   
    I implemented a parser using regular expression (RE) and deterministic finite automaton (DFA) to read input parameters from a file. If you want to modify or improve syntax for parameter file, reimplement using standard tools such as 'bison' and 'flex'. Instead, you can use a 'yaml' or 'json' file as an input parameter file in python scripts.
  
# References
+ T.M. Beardsley, R.K.W. Spencer, and M.W. Matsen, Macromolecules 2019, 52, 8840
+ M.W. Masen, J. Chem. Phys. 2020, 152, 110901
+ T.M. Beardsley, and M.W. Matsen J. Chem. Phys. 2021, 154, 124902
+ M.W. Matsen, and T.M. Beardsley, Polymers 2021, 13, 2437
