# Deep Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) with Deep Learning

# Features
* Diblock Copolymer Melt
* 3D Periodic Boundaries  
* Pseudospectral Implmentation using MKL, FFTW and CUDA
* Accelerating L-FTS using Deep Learning

# Dependencies
#### 1. C++ Compiler
  Any C++ compiler that supports C++11 standard or higher, but I recommend to use Intel compiler. Install Intel oneAPI Base & HPC Toolkit. They are free

#### 2. FFT Library
  The modified diffusion equations are solved by pseudospectral method, and that requires a fast Fourirer transform (FFT) library. You can choose from following FFT libraries.

+ **MKL**   
  Math kernel library (MKL) is bundled with Intel Compilers.  

+ **FFTW**   
  https://www.fftw.org/
  
+ **CUDA**  
  https://developer.nvidia.com/cuda-toolkit  
  
#### 3. OpenMP
  Two partial partition functions are calculated simultaneously using open multi-processing (OpenMP) in the CPU implemenation.  

#### 4. SWIG
  A tool that connects libraries written in C++ with Python    
  http://www.swig.org/

#### 5. PyTorch
  An open source machine learning framwork   
  https://pytorch.org/

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

  Then copy `_langevinfts.so` and `langevinfts.py` to your folder.   
  In python, import the package by adding  `from langevinfts import *`.

* * *
  You can specify your building flags with following command.   
  `cmake ../  -DCMAKE_CXX_COMPILER=[Your CXX Compiler, e.g. "icpc", "g++"]  -DCMAKE_INCLUDE_PATH=[Your FFTW Path]/include -DCMAKE_FRAMEWORK_PATH=[Your FFTW Path]/lib -DUSE_OPENMP=yes`
# User Guide

# Developer Guide
  A few things you need to knows.     

+ **Object Oriented Programming (OOP)**  
    Basic concepts of OOP such as class, inheritance and dynamic binding.   
    In addtion, some design patterns. (class ParamParser, CudaCommon, KernelFactory)

+ **Circular Buffer**  (class CircularBuffer)   
    Circular buffer is a fixed-size buffer that two ends are connected forming circular shape. It is applied in the Anderson mixing iteration to record history of fields in FIFO order.

+ **CUDA Programming** (./platforms/cuda)   
    This is a introductory book written by NVIDIA members  
  https://developer.nvidia.com/cuda-example  
    Optimizing Parallel Reduction in CUDA  
  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf  

+ **Regular Expression (RE)** (class ParamParser)   
    I implemented a parser using RE and deterministic finite automaton (DFA) to read input parameters from file and command line. There is a good online course about RE and finite automaton.  
  https://www.coursera.org/lecture/algorithms-part2/regular-expressions-go3D7 
  
# References
