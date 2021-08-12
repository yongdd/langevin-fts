#CPP=g++
CPP=icpc

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -fno-inline -Wall -traceback -g -debug all

SRC_DIR=../../src
INC_DIR = ../../src/include
CFLAGS+= -I$(INC_DIR)

all : Test

OBJS= SimulationBox.o
TARGET=TestSimulationBox.out

Test: ./TestSimulationBox.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^
SimulationBox.o: $(SRC_DIR)/SimulationBox.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
