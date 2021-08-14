#CPP=g++
CPP=icpc

#CFLAGS=-O3 -Wall -Wextra -pedantic -g -std=c++11
CFLAGS=-O3 -heap-arrays -pg -std=c++11 -fno-inline -Wall -traceback -g -debug all

FFTW_INC=-I/opt/fftw/3.3.8/double/include
FFTW_LIB=-L/opt/fftw/3.3.8/double/lib -lfftw3

SRC_DIR=../../platforms/cpu/src
INC_DIR = ../../platforms/cpu/include

MAIN_SRC_DIR=../../main/src
MAIN_INC_DIR = ../../main/include

CFLAGS+= -I$(INC_DIR) -I$(MAIN_INC_DIR) -I$(FFTW_INC)

all : Test

OBJS=SimulationBox.o CpuSimulationBox.o FftwFFT.o Pseudo.o FftwPseudo.o
TARGET=TestFftwFFT.out

Test: ./TestFftwPseudo.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o $(TARGET) $^ $(FFTW_LIB) -qopenmp
SimulationBox.o: $(MAIN_SRC_DIR)/SimulationBox.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
CpuSimulationBox.o: $(SRC_DIR)/CpuSimulationBox.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^
Pseudo.o: $(MAIN_SRC_DIR)/Pseudo.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^ -qopenmp
FftwFFT.o: $(SRC_DIR)/FftwFFT.cpp
	$(CPP) $(CFLAGS) -o $@ $(FFTW_INC) $(FFTW_LIB) -c $^ -qopenmp
FftwPseudo.o: $(SRC_DIR)/FftwPseudo.cpp
	$(CPP) $(CFLAGS) -o $@ -c $^ -qopenmp
clean:
	rm -f *__*.*
	rm -f *.o
	rm -f *.out
