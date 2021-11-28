# Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS)

# Features
* Diblock Copolymer Melt
* Periodic Boundaries  
* 3D, 2D and 1D (Caution! FTS in 1D and 2D has no physical meaning. They are only for testing methods)
* Pseudospectral Implmentation using MKL, FFTW and CUDA
* Gaussian and Discrete Chain Models

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
+ Intel oneAPI Base & Toolkit 2021.3.0  
+ CUDA Toolkit 11.2  
+ OpenMP bundled with Intel Compilers 2021.3.0  

# Compile
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
  `-DCMAKE_INCLUDE_PATH=[Your FFTW Path]/include \`  
  `-DCMAKE_FRAMEWORK_PATH=[Your FFTW Path]/lib \`  
  `-DUSE_OPENMP=yes`
* * *
  In python, import the package by adding  `from langevinfts import *`.

# User Guide
  Please look around `examples` folder. 

# Cautions  
+ Be aware that unit of length in this program is end-to-end chain length aN^(1/2), not gyration of radius a(N/6)^(1/2), where a is the statistical segment length and N is polymerziation index.  
+ Make sure that this program reproduces the results of previous FTS studies and also produces resonable results. Open source has no warranty.  

# Developer Guide
+ **Abstract Factory**  
    This program is designed to run on different platforms such as FFTW, MKL and CUDA. There is a family of classes for each platform, and `abtract factory pattern` is adopted to produce these classes for given platform.
+ **Anderson Mixing**  
    It is neccesery to store recent history of fields during iteration. For this purpose, it is natural to use `circular buffer` to reduce the number of array copys. If you do not want to use such data structre, please follow the code in [Polymers 2021, 13, 2437]. The performance loss is only marginal.
+ **Reduction in Cuda** (class CudaSimulationBox)   
    Inner product of two fields is caculacted using the CUDA code that NVIDIA provides. If you want to make it simple, please reimplement it using `Cuda Thrust`.
+ **(optional) Parser** (class ParamParser)   
    I implemented a parser using regular expression (RE) and deterministic finite automaton (DFA) to read input parameters from a file. If you want to modify or improve syntax for parameter file, reimplement the parser using standard tools such as `bison` and `flex`. Instead, you can use a `yaml` or `json` file as an input parameter file in python scripts. Using `argparse` is also good option.
  
# References
#### Gaussian Chain Model
+ T.M. Beardsley, R.K.W. Spencer, and M.W. Matsen, Computationally Efficient Field-Theoretic Simulations for Block Copolymer Melts, Macromolecules 2019, 52, 8840   
+ M.W. Masen, Field theoretic approach for block polymer melts: SCFT and FTS, J. Chem. Phys. 2020, 152, 110901   
#### Discrete Chain Model
+ T.M. Beardsley, and M.W. Matsen, Fluctuation correction for the order–disorder transition of diblock copolymer melts, J. Chem. Phys. 2021, 154, 124902   
+ M.W. Matsen, and T.M. Beardsley, Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation, Polymers 2021, 13, 2437   
####  Field-Update Algorithms
+ D.L. Vigil, K.T. Delaney, and G.H. Fredrickson, Quantitative Comparison of Field-Update Algorithms for Polymer SCFT and FTS, Macromolecules 2021, 54, 21, 9804
