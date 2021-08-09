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

all : test1 

#------------ Random Number ----------------

OBJS1=constants.o random_gaussian.o
TAR1=random_test.out
test1: ./random_test.f90 $(OBJS1)
	$(FC) $(FFLAGS) -o $(TAR1)  $^

#------------ Dependence ------------------

constants.o: $(SRCDIR)/constants.F90
	$(FC) $(FFLAGS) -o $@ -c $^
random_gaussian.o: $(SRCDIR)/random_gaussian.F90
	$(FC) $(FFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.mod
	rm -f fields_* *.txt
	rm -f *.out
