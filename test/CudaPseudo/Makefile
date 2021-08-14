#CPP=g++
CPP=icpc
NVCC=nvcc 

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -std=c++11 -fno-inline -Wall -traceback -g -debug all
NFLGAS=-O3 --debug --std c++11 -Werror -L/usr/local/cuda/lib64 -lcufft -lcudart

SRC_DIR=../../platforms/cuda/src
INC_DIR = ../../platforms/cuda/include

CFLAGS+= -I$(INC_DIR)
NFLAGS+= -I$(INC_DIR)

all : Test

OBJS= CudaSimulationBox.o CudaCommon.o CudaPseudo.o 
TARGET=TestCudaPseudo.out

Test: ./TestCudaPseudo.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^ $(NFLGAS) -lstdc++
CudaSimulationBox.o: $(SRC_DIR)/CudaSimulationBox.cpp
	$(NVCC) $(NFLAGS) -o $@ -c $^
CudaCommon.o: $(SRC_DIR)/CudaCommon.cu
	$(NVCC) $(NFLAGS)  -o $@ -c $^
CudaPseudo.o: $(SRC_DIR)/CudaPseudo.cu
	$(NVCC) $(NFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
