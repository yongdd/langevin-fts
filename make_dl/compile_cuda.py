#!python.exe
# -*- coding: UTF-8 -*-
import os

USE_SINGLE_PRECISION=0
f = open('.f2py_f2cmap', 'w')
if (USE_SINGLE_PRECISION==0):
   f.write("dict(real=dict(sp='float', dp='double', rp='double'))") 
else:
   f.write("dict(real=dict(sp='float', dp='double', rp='float'))")
f.close()

# preprocessor options
preprocessors = "-DUSE_SINGLE_PRECISION=" + str(USE_SINGLE_PRECISION) + " " 
'''comepile CUDA libarary'''
cuda_exe = "nvcc -c -O3 -Xcompiler -fPIC "
cuda_libs="-L/opt/cuda/lib64 -lcufft -lcudart "

cuda_sources = ("../src/cuda/pseudo.cu "
                "../src/cuda/anderson_mixing.cu "
                )

cuda_cmd = cuda_exe + cuda_libs + preprocessors + cuda_sources

'''Uses f2py to compile needed library'''
# build command-strings
# command for compling *.o and *.mod files
fortran_exe = "mpiifort -c -O3 -fPIC -heap-arrays "

# fortran compiler settings
fortran_flags = " " #-pg -fno-inline #  -g -Wall -Wextra -std=f2003 -pedantic -fimplicit-none -fcheck=all -fbacktrace "
fortran_flags += preprocessors + " "
mpi_path = "/opt/intel/mpi/latest/"
mpi_libs = "-I" + mpi_path + "include -L" + mpi_path + "lib/release -lmpi -L" + mpi_path + "lib -lmpifort "

# add path to source code
fortran_sources = ("../src/constants.F90 "
                  "../src/param_parser.F90 "
                  "../src/simulation_box.F90 "
                  "../src/polymer_chain.F90 "
                  "../src/parallel_tempering.F90 "
                  "../src/cuda/pseudo_wrapper.F90 "
                  "../src/cuda/anderson_mixing_wrapper.F90 "
                  "../src/random_gaussian.F90 "
                  "../src/langevin_equation.F90 ")
                  
# assemble fortran command
fortran_cmd = fortran_exe +  mpi_libs + fortran_flags + fortran_sources

# command for compiling main source file using f2py
f2py_exe = "f2py -c --f90exec='mpiifort' "
f2py_flags = "--compiler=unix --fcompiler=intelem -m lfts_fortran "

# add path to source code/ dependencies
f2py_sources = ("../src/lfts.F90 "
                 "constants.o "
                 "param_parser.o "
                 "simulation_box.o "
                 "polymer_chain.o "
                 "parallel_tempering.o "
                 "pseudo.o pseudo_wrapper.o "
                 "anderson_mixing.o anderson_mixing_wrapper.o "
                 "langevin_equation.o " 
                 "random_gaussian.o ")
# assemble f2py command
f2py_cmd = f2py_exe + f2py_flags + preprocessors + cuda_libs + mpi_libs + f2py_sources

# compile .o and .mod files
print ("compiling object- and module-files...\n")
print (cuda_cmd)
os.system(cuda_cmd)
print (fortran_cmd)
os.system(fortran_cmd)
# compile main_source.f90 with f2py
print ("================================================================")
print ("start f2py...")
print
print (f2py_cmd)
os.system(f2py_cmd)
