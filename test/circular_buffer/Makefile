USE_SINGLE_PRECISION=0 ## 1: SINGLE or 0: DOUBLE

#FC=gfortran
FC=ifort

#FFLAGS=-O3 
#FFLAGS=-O3 -Wall -Wextra -std=f2003 -pedantic -fimplicit-none -fcheck=all -fbacktrace -g

#FFLAGS=-O3 -heap-arrays
#FFLAGS=-O3 -heap-arrays -pg -fno-inline
FFLAGS=-O3 -heap-arrays -pg -fno-inline -warn -stand f03 -CB -traceback -g -debug all

PPFLAGS = -DUSE_SINGLE_PRECISION=$(USE_SINGLE_PRECISION)
FFLAGS += $(PPFLAGS)

SRCDIR=../../src
##TESTDIR=../../src/test

all : test

#------------ Circular Buffer -----------

OBJS=constants.o circular_buffer.o
TAR=circular_buffer_test.out
test: ./circular_buffer_test.f90 $(OBJS)
	$(FC) $(FFLAGS) -o $(TAR)  $^

#------------ Dependence ------------------

constants.o: $(SRCDIR)/constants.F90
	$(FC) $(FFLAGS) -o $@ -c $^
circular_buffer.o: $(SRCDIR)/mkl/circular_buffer.F90
	$(FC) $(FFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.mod
	rm -f fields_* *.txt
	rm -f *.out
