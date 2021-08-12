#CPP=g++
CPP=icpc

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -fno-inline -Wall -traceback -g -debug all

SRC_DIR=../../src
INC_DIR = ../../src/include
CFLAGS+= -I$(INC_DIR)

all : Test

OBJS= ParamParser.o
TARGET=TestParamParser.out

Test: ./TestParamParser.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^
ParamParser.o: $(SRC_DIR)/ParamParser.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
