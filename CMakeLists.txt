# CMake
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(FTS_DEEP_LEARNING CXX CUDA)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CUDA_STANDARD 11)
SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug  )

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

#---------- FTS_LIBRARIES -----------

#  OpenMP
IF(USE_OPENMP)
	FIND_PACKAGE(OpenMP QUIET)
	IF (OPENMP_FOUND)
		SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	ENDIF()
ENDIF()

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
	main/src/RandomGaussian.cpp
	)

#  Intel MKL
FIND_PACKAGE(MKL QUIET)
IF(MKL_FOUND AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	SET(BUILD_CPU_MKL_LIB TRUE)
ELSE()
	SET(BUILD_CPU_MKL_LIB FALSE)
ENDIF()
IF(BUILD_CPU_MKL_LIB)
	ADD_DEFINITIONS(-DUSE_CPU_MKL)
	ADD_LIBRARY(cpu-mkl
        platforms/cpu/src/MklFFT.cpp
	platforms/cpu/src/CpuPseudo.cpp
        platforms/cpu/src/CpuAndersonMixing.cpp
	)
ENDIF()

#  FFTW 
FIND_PATH(FFTW_INCLUDE NAMES fftw3.h PATHS /opt/fftw/3.3.8/double/include)
FIND_LIBRARY(FFTW_LIBRARY NAMES fftw3 PATHS /opt/fftw/3.3.8/double/lib)
MESSAGE("${FFTW_INCLUDE}")
MESSAGE("${FFTW_LIBRARY}")
IF(FFTW_INCLUDE AND FFTW_LIBRARY)
	SET(BUILD_CPU_FFTW_LIB TRUE)
ELSE()
	SET(BUILD_CPU_FFTW_LIB FALSE)
ENDIF()
IF(BUILD_CPU_FFTW_LIB)
	ADD_DEFINITIONS(-DUSE_CPU_FFTW)
        ADD_LIBRARY(cpu-fftw
	platforms/cpu/src/FftwFFT.cpp
	platforms/cpu/src/CpuPseudo.cpp
        platforms/cpu/src/CpuAndersonMixing.cpp
        )
ENDIF()

# NVIDIA CUDA
FIND_PACKAGE(CUDAToolkit REQUIRED)
IF(CUDAToolkit_FOUND)
	ADD_DEFINITIONS(-DUSE_CUDA)
	ADD_LIBRARY(cuda
        platforms/cuda/src/CudaCommon.cu
        platforms/cuda/src/CudaSimulationBox.cu
        platforms/cuda/src/CudaPseudo.cu
        platforms/cuda/src/CudaAndersonMixing.cu
	)
ENDIF()

IF( (NOT BUILD_CPU_MKL_LIB) AND
    (NOT BUILD_CPU_FFTW_LIB) AND
    (NOT CUDAToolkit_FOUND) )
    MESSAGE( FATAL_ERROR "Could not find any FFT library, CMake will exit." )
ENDIF()

#  Factory
ADD_LIBRARY(factory
	main/src/KernelFactory.cpp
        )

#---------- EXECUTABLE -----------

ADD_EXECUTABLE(scft.out main/SCFT.cpp)

TARGET_LINK_LIBRARIES(scft.out
	factory
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,cuda,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cuda_driver,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cufft,>
	$<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
	$<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-mkl,>
	$<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,cpu-fftw,>
	$<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,${FFTW_LIBRARY},>	
	common)

#---------- Python Wrapper -----------

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

MESSAGE("${Python3_EXECUTABLE}")
MESSAGE("${Python3_INCLUDE_DIRS}")
MESSAGE("${Python3_NumPy_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES(${Python3_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Python3_NumPy_INCLUDE_DIRS})

SET_SOURCE_FILES_PROPERTIES(swig/LangevinFTS.i PROPERTIES CPLUSPLUS ON)
SWIG_ADD_LIBRARY(langevin_fts
        LANGUAGE python
        SOURCES swig/LangevinFTS.i)

SWIG_LINK_LIBRARIES(langevin_fts
        factory
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,cuda,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cuda_driver,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cufft,>
        $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
        $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-mkl,>
        $<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,cpu-fftw,>
        $<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,${FFTW_LIBRARY},>
        common)

#---------- TEST -----------

ENABLE_TESTING()
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/tests)
