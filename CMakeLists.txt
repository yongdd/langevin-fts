# CMake

SET(CMAKE_C_COMPILER "icc")
SET(CMAKE_CXX_COMPILER "icpc")

CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(FTS_DEEP_LEARNING CXX CUDA)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(CMAKE_CUDA_STANDARD 11)
SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

SET ( CMAKE_BUILD_TYPE Debug )
#SET ( CMAKE_BUILD_TYPE Release )

##------------------------------------

# Openmp
FIND_PACKAGE(OpenMP REQUIRED)

# Math Kernel Library
FIND_PACKAGE(MKL REQUIRED)

# CUDA
FIND_PACKAGE(CUDA REQUIRED)

## SWIG
#FIND_PACKAGE(SWIG REQUIRED)

## Python
#FIND_PACKAGE(PythonInterp 3.5 REQUIRED)

##------------------------------------

##ADD_SUBDIRECTORY(src)


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/main/include/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/platforms/cpu/include/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/platforms/cuda/include/)

ADD_LIBRARY(main
	main/src/ParamParser.cpp
	main/src/PolymerChain.cpp)

ADD_LIBRARY(cpu
	platforms/cpu/src/CpuSimulationBox.cpp
	platforms/cpu/src/MklFFT.cpp
	platforms/cpu/src/MklPseudo.cpp
	platforms/cpu/src/CpuCircularBuffer.cpp
	platforms/cpu/src/CpuAndersonMixing.cpp
	)

ADD_LIBRARY(cuda
	platforms/cuda/src/CudaCommon.cu
        platforms/cuda/src/CudaSimulationBox.cu
	platforms/cuda/src/CudaPseudo.cu
	platforms/cpu/src/CpuCircularBuffer.cpp
        platforms/cuda/src/CudaAndersonMixing.cu
	)

##LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

#LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/)
#LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/platforms/cuda/)

ADD_EXECUTABLE(scft.out main/SCFT.cpp)
TARGET_LINK_LIBRARIES(scft.out PUBLIC main cpu cuda -mkl -lcufft -lcudart)
