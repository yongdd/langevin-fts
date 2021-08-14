#CPP=g++
CPP=icpc

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -fno-inline -Wall -traceback -g -debug all

SRC_DIR=../../platforms/cpu/src
INC_DIR=../../platforms/cpu/include

MAIN_SRC_DIR=../../main/src
MAIN_INC_DIR=../../main/include

CFLAGS+= -I$(INC_DIR) -I$(MAIN_INC_DIR)

all : Test

OBJS=SimulationBox.o AndersonMixing.o \
     CpuSimulationBox.o CpuCircularBuffer.o CpuAndersonMixing.o
TARGET=TestAndersonMixing.out

Test: ./TestAndersonMixing.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^
SimulationBox.o: $(MAIN_SRC_DIR)/SimulationBox.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
AndersonMixing.o: $(MAIN_SRC_DIR)/AndersonMixing.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
CpuSimulationBox.o: $(SRC_DIR)/CpuSimulationBox.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
CpuCircularBuffer.o: $(SRC_DIR)/CpuCircularBuffer.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
CpuAndersonMixing.o: $(SRC_DIR)/CpuAndersonMixing.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
