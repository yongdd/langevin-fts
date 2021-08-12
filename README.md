# Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) with Deep Learning

# Features
* 3D periodic boundaries  
* Pseudospectral implmentation using CUDA
* Accelerating L-FTS using Deep Learning

# Dependencies
#### 1. C++ Compiler
  A C++ compiler that supports C++11 standard or higher.

#### 2. Fast Fourirer Transform(FFT) Library
  The modified diffusion equations are solved by pseudospectral method, and that requires a FFT library. You can choose one of the following FFT libraries.

+ **MKL**   
  MKL is bundled with Intel Compilers.  
  
+ **CUDA**  
  https://developer.nvidia.com/cuda-toolkit  
  
#### 3. Open Multi-Processing(OpenMP)
  Two partial partition functions are calculated simultaneously using OpenMP in implemenation.  
#### 4. SWIG

#### 5. PyTorch

* * *
I tested this program under following environments.  
+ Fortran Compilers
  + Intel Fortran Compiler 2021.3.0  
+ CUDA Toolkit 11.2
+ MPI/OpenMP bundled with Intel Compilers 2021.3.0

# Compile
  
# User Guide

# Developer Guide
  I tried to keep each part simple as long as it is not critical to performance. However, some parts of this code might still be hard for you unless you studied relevant subjects.   

+ **Circular Buffer**   
    Circular buffer is a fixed-size buffer that two ends are connected forming circular shape. It is applied in the Anderson mixing iteration to record history of fields in FIFO order.

+ **CUDA Programming**  
    This is a introductory book written by NVIDIA members  
  https://developer.nvidia.com/cuda-example  
    Optimizing Parallel Reduction in CUDA  
  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf  

+ **Regular Expression(RE)**  
    I implemented a parser using RE and deterministic finite automaton (DFA) to read input parameters from file and command line. There is a good online course about RE and finite automaton.  
  https://www.coursera.org/lecture/algorithms-part2/regular-expressions-go3D7 
  
# References
