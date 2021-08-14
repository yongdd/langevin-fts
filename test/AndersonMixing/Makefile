#CPP=g++
CPP=icpc

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -fno-inline -Wall -traceback -g -debug all

SRC_DIR=../../platforms/cpu/src
INC_DIR=../../platforms/cpu/include

CFLAGS+= -I$(INC_DIR)

all : Test

OBJS=SimulationBox.o CircularBuffer.o AndersonMixing.o
TARGET=TestAndersonMixing.out

Test: ./TestAndersonMixing.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^
SimulationBox.o: $(SRC_DIR)/SimulationBox.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
CircularBuffer.o: $(SRC_DIR)/CircularBuffer.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
AndersonMixing.o: $(SRC_DIR)/AndersonMixing.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
