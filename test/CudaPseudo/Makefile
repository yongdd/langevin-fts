#CPP=g++
CPP=icpc
NVCC=nvcc 

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -std=c++11 -fno-inline -Wall -traceback -g -debug all
NFLGAS=-O3 --debug --std c++11 -Werror -L/usr/local/cuda/lib64 -lcufft -lcudart

SRC_DIR=../../src
INC_DIR = ../../src/include

CUDA_SRC_DIR=../../src/platforms/cuda
CUDA_INC_DIR = ../../src/platforms/cuda/include

CFLAGS+= -I$(INC_DIR) -I$(CUDA_INC_DIR)
NFLAGS+= -I$(INC_DIR) -I$(CUDA_INC_DIR)

all : Test

OBJS= CudaCommon.o CudaPseudo.o 
TARGET=TestCudaPseudo.out

Test: ./TestCudaPseudo.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^ $(NFLGAS) -lstdc++
CudaCommon.o: $(CUDA_SRC_DIR)/CudaCommon.cu
	$(NVCC) $(NFLAGS)  -o $@ -c $^
CudaPseudo.o: $(CUDA_SRC_DIR)/CudaPseudo.cu
	$(NVCC) $(NFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
