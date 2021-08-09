# Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) with Deep Learning

# Features
* 3D periodic boundaries  
* Pseudospectral implmentation using CUDA
* Accelerating L-FTS using Deep Learning

# Dependencies
#### 1. Fortran Compiler
  A fortran compiler that supports 2003 standard or higher, and this program requires a preprocessor to choose floating point precision in the compile time. The preprocesor is non-standard feature in Fortran, but many fortran compilers provide it. It is automatically enabled if the file extension is capitalized, i.e., .F90, or .F03.

#### 2. Fast Fourirer Transform(FFT) Library
  The modified diffusion equations are solved by pseudospectral method, and that requires a FFT library. You can choose one of the following FFT libraries.

+ **MKL**   
  MKL is bundled with Intel Compilers.  
  
+ **CUDA**  
  https://developer.nvidia.com/cuda-toolkit  

#### 3. Message Passing Interface(MPI)
  Parallel tempering is implemented using MPI.  
  
#### 4. Open Multi-Processing(OpenMP)
  Two partial partition functions are calculated simultaneously using OpenMP in FFTW and MKL implemenation.  

#### 5. f2py

#### 6. PyTorch

* * *
I tested this program under following environments.  
+ Fortran Compilers
  + GNU Fortran compiler 7.5  
  + Intel Fortran Compiler 19.1  
  + AMD Optimizing C/C++ Compiler 2.1  
+ FFTW 3.3.8
+ CUDA Toolkit 11.0
+ MPI/OpenMP bundled with Intel Compilers 19.1

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
