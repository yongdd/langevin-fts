#CPP=g++
CPP=icpc

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -fno-inline -Wall -traceback -g -debug all

COMMON_SRC_DIR=../../src/platforms/common
COMMON_INC_DIR=../../src/platforms/common/include

SRC_DIR=../../src/platforms/mkl
INC_DIR=../../src/platforms/mkl/include

CFLAGS+= -I$(COMMON_INC_DIR) -I$(INC_DIR)

all : Test

OBJS=CircularBuffer.o MklAndersonMixing.o
TARGET=TestAndersonMixing.out

Test: ./TestAndersonMixing.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^
CircularBuffer.o: $(COMMON_SRC_DIR)/CircularBuffer.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
MklAndersonMixing.o: $(SRC_DIR)/MklAndersonMixing.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
