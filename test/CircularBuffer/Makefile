#CPP=g++
CPP=icpc

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -fno-inline -Wall -traceback -g -debug all

SRC_DIR=../../platforms/cpu/src
INC_DIR = ../../platforms/cpu/include
CFLAGS+= -I$(INC_DIR)

all : Test

OBJS= CircularBuffer.o
TARGET=TestCircularBuffer.out

Test: ./TestCircularBuffer.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^
CircularBuffer.o: $(SRC_DIR)/CircularBuffer.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
