import os

# Preprocessor options
preprocessors = " "

# CUDA command
cuda_exe = "nvcc -c -O3 -Xcompiler -fPIC "
cuda_libs="-L/opt/cuda/lib64 -lcufft -lcudart "

cuda_sources = ("../src/cuda/pseudo.cu "
                "../src/cuda/anderson_mixing.cu "
                )

cuda_cmd = cuda_exe + cuda_libs + preprocessors + cuda_sources

# Fortran command for compling *.o and *.mod files
fortran_exe = "mpiifort -c -O3 -fPIC -heap-arrays "
fortran_flags = " " #-pg -fno-inline #  -g -Wall -Wextra -std=f2003 -pedantic -fimplicit-none -fcheck=all -fbacktrace "
fortran_flags += preprocessors + " "
mpi_path = "/opt/intel/mpi/latest/"
mpi_libs = "-I" + mpi_path + "include -L" + mpi_path + "lib/release -lmpi -L" + mpi_path + "lib -lmpifort "
fortran_sources = (
                  "../src/param_parser.F90 "
                  "../src/simulation_box.F90 "
                  "../src/polymer_chain.F90 "
                  "../src/parallel_tempering.F90 "
                  "../src/cuda/pseudo_wrapper.F90 "
                  "../src/cuda/anderson_mixing_wrapper.F90 "
                  "../src/random_gaussian.F90 "
                  "../src/langevin_equation.F90 "
                  )
fortran_cmd = fortran_exe +  mpi_libs + fortran_flags + fortran_sources

# Command for compiling main source file using f2py
f2py_exe = "f2py -c --f90exec='mpiifort' "
f2py_flags = "--compiler=unix --fcompiler=intelem -m lfts_fortran "
f2py_sources = ("../src/lfts.F90 "
                 "param_parser.o "
                 "simulation_box.o "
                 "polymer_chain.o "
                 "parallel_tempering.o "
                 "pseudo.o pseudo_wrapper.o "
                 "anderson_mixing.o anderson_mixing_wrapper.o "
                 "langevin_equation.o " 
                 "random_gaussian.o ")
f2py_cmd = f2py_exe + f2py_flags + preprocessors + cuda_libs + mpi_libs + f2py_sources

# compile .cu files
print ("compiling CUDA files...")
print (cuda_cmd)
os.system(cuda_cmd)
# compile .o and .mod files
print ("\ncompiling object- and module-files...")
print (fortran_cmd)
os.system(fortran_cmd)
# compile main_source.f90 with f2py
print ("\n===============================================")
print ("start f2py...")
print (f2py_cmd)
os.system(f2py_cmd)
