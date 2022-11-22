# Langevin FTS
Langevin Field-Theoretic Simulation (L-FTS) for Python

# Features
* SCFT and L-FTS   
* Linear AB Copolymer Melts  
* Polymer Mixture (**beta**)  
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
  Any C++ compiler that supports C++14 standard or higher. To use MKL, install Intel oneAPI toolkit.

#### CUDA Toolkit
  https://developer.nvidia.com/cuda-toolkit   
  Required for the GPU computation. If it is not installed, ask admin for its installation.

#### Anaconda
  https://www.anaconda.com/

* * *
Environment variables must be set so that `nvcc` and `conda` can be executed in the command line (Type `which nvcc` and `which conda` to check the installation).


# Compiling
```Shell
# Create virtual environment 
conda create -n lfts python=3.9 cmake=3.19 conda git \
      pybind11=2.9 scipy openmpi pyyaml  
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
* If you want to use FFTW library, clone from `fftw` branch. That is distributed under GPL license.  
```Shell
git clone -b fftw https://github.com/yongdd/langevin-fts.git  
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
+ This is not an application but a library for SCFT and L-FTS, and you need to write your own program using Python language. It requires a little programming, but this approach provides flexibility and you can easily customize your applications.   
+ To understand the entire process of simulations, please refer to sample scripts in `example/scft_single_file` and `example/fts_single_file`, and use sample scripts in the `example/scft` and `example/fts` to perform actual simulations.   
+ If your ultimate goal is to use deep learning boosted L-FTS, you may use the sample scripts of DL-FTS repository. (One can easily turn on/off deep learning from the scripts.)  
+ To use this library, first activate virtual environment by typing `conda activate lfts` in command line. In Python script, import the package by adding  `from langevinfts import *`.   
+ Be aware that the unit of length in this library is the end-to-end chain length *aN^(1/2)*, not the gyration of radius *a(N/6)^(1/2)*, where *a* is reference statistical segment length and *N* is reference polymerization index.  
+ The fields acting on chain are described using `per chain` language instead of `per segment` language for both SCFT and L-FTS. The same notation is used in [*Macromolecules* **2013**, 46, 8037]. If you want to obtain the same fields used in [*Polymers* **2021**, 13, 2437], multiply *1/N* to each field.
+ Please refer to [*J. Chem. Phys.* **2014**, 141, 174103] for how to perform polymer field theory simulations with polymer mixtures.
+ Use FTS in 1D and 2D only for the test. It does not have a physical meaning.
+ Open-source has no warranty. Make sure that this program reproduces the results of previous FTS studies, and also produces reasonable results.  
+ Matlab and Python tools for visualization and renormalization are included in `tools` folder.   

# Developer Guide
#### Platforms  
  This program is designed to run on different platforms such as MKL and CUDA, and there is a family of classes for each platform. To produce instances of these classes for given platform, `abstract factory pattern` is adopted.   

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