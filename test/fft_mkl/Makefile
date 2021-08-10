USE_SINGLE_PRECISION=0 ## 1: SINGLE or 0: DOUBLE

FC=ifort

#FFLAGS=-O3 -heap-arrays
#FFLAGS=-O3 -heap-arrays -pg -fno-inline
FFLAGS=-O3 -heap-arrays -pg -fno-inline -warn -stand f03 -CB -traceback -g -debug all

PPFLAGS = -DUSE_SINGLE_PRECISION=$(USE_SINGLE_PRECISION)
FFLAGS += $(PPFLAGS)

SRCDIR=../../src

all : test3

#------------ FFT MKL 3D  ----------------

OBJS3= fft_mkl.o mkl_dfti.o mkl_trig_transforms.o
TAR3=fft_mkl_3d_test.out
test3: ./fft_mkl_3d_test.f90 $(OBJS3)
	$(FC) $(FFLAGS) -o $(TAR3)  $^ -mkl -qopenmp

#------------ Dependence ------------------

fft_mkl.o: $(SRCDIR)/mkl/fft_mkl.F90 mkl_dfti.o mkl_trig_transforms.o
	$(FC) $(FFLAGS) -o $@ -c -mkl -qopenmp $(SRCDIR)/mkl/fft_mkl.F90
mkl_dfti.o: /opt/intel/mkl/latest/include/mkl_dfti.f90
	$(FC) $(FFLAGS) -o $@ -c -mkl $^ 
mkl_trig_transforms.o: /opt/intel/mkl/latest/include/mkl_trig_transforms.f90
	$(FC) $(FFLAGS) -o $@ -c -mkl $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.mod
	rm -f fields_* *.txt
	rm -f *.out
