#CPP=g++
CPP=icpc

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -std=c++11 -fno-inline -Wall -traceback -g -debug all

SRC_DIR=../../platforms/cpu/src
INC_DIR = ../../platforms/cpu/include

CFLAGS+= -I$(INC_DIR)
NFLAGS+= -I$(INC_DIR)

all : Test

OBJS=MklFFT.o
TARGET=TestMklFFT.out

Test: ./TestMklFFT.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^ -mkl -qopenmp
MklFFT.o: $(SRC_DIR)/MklFFT.cpp
	$(CPP) $(CFLAGS) -o $@ -c -mkl -qopenmp $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
