USE_SINGLE_PRECISION=0 ## 1: SINGLE or 0: DOUBLE

FC=gfortran
#FC=ifort
NVCC=nvcc

#FFLAGS=-O3 -cpp
FFLAGS=-O3 -cpp -Wall -Wextra -std=f2003 -pedantic -fimplicit-none -fcheck=all -fbacktrace -g

#FFLAGS=-O3 -fpp -heap-arrays
#FFLAGS=-O3 -fpp -heap-arrays -pg -fno-inline
#FFLAGS=-O3 -fpp -heap-arrays -pg -fno-inline -warn -stand f03 -CB -traceback -g -debug all

CUDAFLAGS=-O3 -L/usr/local/cuda/lib64 -lcufft -lcudart

PPFLAGS = -DUSE_SINGLE_PRECISION=$(USE_SINGLE_PRECISION)
FFLAGS += $(PPFLAGS)
CUDAFLAGS += $(PPFLAGS)

SRCDIR=../../src

all : test3

#------------ Pseudo GPU  ----------------

OBJS3=pseudo_gpu_cuda.o
TAR3=pseudo_gpu_test.out
test3: ./pseudo_gpu_test.f90 $(OBJS3)
	$(FC) $(FFLAGS) -o $(TAR3)  $^ $(CUDAFLAGS) -lstdc++

#------------ Dependence ------------------
pseudo_gpu_cuda.o: $(SRCDIR)/cuda/pseudo.cu
	$(NVCC) $(CUDAFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.mod
	rm -f fields_* *.txt
	rm -f *.out
