#CPP=g++
CPP=icpc

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -std=c++11 -fno-inline -Wall -traceback -g -debug all

SRC_DIR=../../src
INC_DIR = ../../src/include

MKL_SRC_DIR=../../src/platforms/mkl
MKL_INC_DIR = ../../src/platforms/mkl/include

CFLAGS+= -I$(INC_DIR) -I$(MKL_INC_DIR)
NFLAGS+= -I$(INC_DIR) -I$(MKL_INC_DIR)

all : Test

OBJS=MklFFT.o MklPseudo.o
TARGET=TestMklFFT.out

Test: ./TestMklPseudo.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^ -mkl -qopenmp
MklFFT.o: $(MKL_SRC_DIR)/MklFFT.cpp
	$(CPP) $(CFLAGS) -o $@ -c -mkl -qopenmp $^
MklPseudo.o: $(MKL_SRC_DIR)/MklPseudo.cpp
	$(CPP) $(CFLAGS) -o $@ -c -mkl -qopenmp $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
