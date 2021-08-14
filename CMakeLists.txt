# CMake

CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(FTS_DEEP_LEARNING CXX CUDA)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CUDA_STANDARD 11)
SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

#SET ( CMAKE_BUILD_TYPE Debug )
SET ( CMAKE_BUILD_TYPE Release )

#---------- FTS_LIBRARIES -----------
#FIND_PACKAGE(OpenMP QUIET)
if (OPENMP_FOUND)
	SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#FIND_PACKAGE(SWIG REQUIRED)
#FIND_PACKAGE(PythonInterp 3.5 REQUIRED)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/main/include/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/platforms/cpu/include/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/platforms/cuda/include/)

#  Common Files

ADD_LIBRARY(common
	main/src/ParamParser.cpp
	main/src/PolymerChain.cpp
	main/src/SimulationBox.cpp
	main/src/CircularBuffer.cpp
	main/src/Pseudo.cpp
	main/src/AndersonMixing.cpp
	)

#  Intel MKL
FIND_PACKAGE(MKL QUIET)
IF(MKL_FOUND AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	SET(BUILD_CPU_MKL_LIB TRUE)
ELSE()
	SET(BUILD_CPU_MKL_LIB FALSE)
ENDIF()
IF(BUILD_CPU_MKL_LIB)
	ADD_LIBRARY(cpu-mkl
        platforms/cpu/src/CpuSimulationBox.cpp
        platforms/cpu/src/MklFFT.cpp
	platforms/cpu/src/MklPseudo.cpp
        platforms/cpu/src/CpuAndersonMixing.cpp
	)
ENDIF()

#  FFTW 
FIND_PATH (FFTW_INCLUDE fftw3.h /opt/fftw/3.3.8/double/include)
FIND_LIBRARY (FFTW_LIBRARY NAMES fftw3 /opt/fftw/3.3.8/double/lib)
MESSAGE (${FFTW_INCLUDE})
MESSAGE (${FFTW_LIBRARY})
IF(FFTW_INCLUDE AND FFTW_LIBRARY)
	SET(BUILD_CPU_FFTW_LIB TRUE)
ELSE()
	SET(BUILD_CPU_FFTW_LIB FALSE)
ENDIF()
IF(BUILD_CPU_FFTW_LIB)
        ADD_LIBRARY(cpu-fftw
        platforms/cpu/src/CpuSimulationBox.cpp
	platforms/cpu/src/FftwFFT.cpp
	platforms/cpu/src/FftwPseudo.cpp
        platforms/cpu/src/CpuAndersonMixing.cpp
        )
ENDIF()

# NVIDIA CUDA
FIND_PACKAGE(CUDAToolkit REQUIRED)
IF(CUDAToolkit_FOUND)
	ADD_LIBRARY(cuda
        platforms/cuda/src/CudaCommon.cu
        platforms/cuda/src/CudaSimulationBox.cu
        platforms/cuda/src/CudaPseudo.cu
        platforms/cuda/src/CudaAndersonMixing.cu
	)
ENDIF()

IF( (NOT MKL_FOUND) AND (NOT CUDAToolkit_FOUND) )
	MESSAGE( FATAL_ERROR "Cannot find any libraries, CMake will exit." )
ENDIF()

#---------- EXECUTABLE -----------

#LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/)
ADD_EXECUTABLE(scft.out main/SCFT.cpp)

TARGET_LINK_LIBRARIES(scft.out
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,cuda,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cuda_driver,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cufft,>
	$<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
	$<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-mkl,>
	$<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,cpu-fftw,>
	$<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,${FFTW_LIBRARY},>	
	common)

