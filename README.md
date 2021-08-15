# Deep Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) with Deep Learning

# Features
* Diblock Copolymer Melt
* 3D periodic boundaries  
* Pseudospectral implmentation using MKL, FFTW and CUDA
* Accelerating L-FTS using Deep Learning

# Dependencies
#### 1. C++ Compiler
  Any C++ compiler that supports C++11 standard or higher, but I recommend to use Intel compiler. Install Intel oneAPI Base & HPIC Toolkit. They are free.

#### 2. Fast Fourirer Transform(FFT) Library
  The modified diffusion equations are solved by pseudospectral method, and that requires a FFT library. You can choose one of the following FFT libraries.

+ **MKL**   
  MKL is bundled with Intel Compilers.  

+ **FFTW**   
  Fastest Fourier transform in the West  
  https://www.fftw.org/
  
+ **CUDA**  
  https://developer.nvidia.com/cuda-toolkit  
  
#### 3. Open Multi-Processing(OpenMP)
  Two partial partition functions are calculated simultaneously using OpenMP in the CPU implemenation.  

#### 4. SWIG

#### 5. PyTorch

* * *
I tested this program under following environments.  
+ C/C++ Compilers
  + Intel oneAPI Base & Toolkit 2021.3.0   
  + The GNU Compiler Collection 7.5 
+ CUDA Toolkit 11.2
+ MPI/OpenMP bundled with Intel Compilers 2021.3.0

# Compile
  
  git clone https://github.com/yongdd/Langevin_FTS_Public.git  
  mkdir build  
  cd build  
  cmake <Options> ../  
  make  

+  Available Options

  + To change compilers
    -DCMAKE_CXX_COMPILER=<Your CXX Compiler, e.g. "icpc", "g++">
  + To use FFTW  
    -DCMAKE_INCLUDE_PATH=<Your FFTW_PATH>/include -DCMAKE_FRAMEWORK_PATH=<Your FFTW_PATH>/lib
  + To use OpenMP  
    -DUSE_OPENMP  

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
