# CMake
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(LANGEVIN_FTS CXX)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CUDA_STANDARD 11)
SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug  )

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
#---------- FTS_LIBRARIES -----------
IF(CMAKE_BUILD_TYPE STREQUAL Debug)
    SET(CMAKE_VERBOSE_MAKEFILE ON)
    ADD_COMPILE_OPTIONS(-Wall -Wextra)
ELSE()
    ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
ENDIF()

# OpenMP
FIND_PACKAGE(OpenMP)
IF (OPENMP_FOUND)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/common/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/platforms/cpu/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/platforms/cuda/)

# Common Files
ADD_LIBRARY(common
    src/common/ParamParser.cpp
    src/common/PolymerChain.cpp
    src/common/SimulationBox.cpp
    src/common/CircularBuffer.cpp
    src/common/Pseudo.cpp
    src/common/AndersonMixing.cpp
)

# Intel MKL
FIND_PACKAGE(MKL)
IF(MKL_FOUND AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    SET(BUILD_CPU_MKL_LIB TRUE)
    ADD_DEFINITIONS(-DUSE_CPU_MKL)
    ADD_LIBRARY(cpu-mkl
        src/platforms/cpu/MklFFT1D.cpp
        src/platforms/cpu/MklFFT2D.cpp
        src/platforms/cpu/MklFFT3D.cpp
    	src/platforms/cpu/CpuPseudoContinuous.cpp
        src/platforms/cpu/CpuPseudoDiscrete.cpp
    	src/platforms/cpu/CpuAndersonMixing.cpp
        src/platforms/cpu/MklFactory.cpp
    )
ELSE()
    SET(BUILD_CPU_MKL_LIB FALSE)
ENDIF()

# Pocket FFT
SET(BUILD_CPU_POCKET_FFT_LIB TRUE)
ADD_DEFINITIONS(-DUSE_CPU_POCKET_FFT)
ADD_DEFINITIONS(-DPOCKETFFT_NO_MULTITHREADING)
ADD_LIBRARY(cpu-pocketfft
    src/platforms/cpu/PocketFFT1D.cpp
    src/platforms/cpu/PocketFFT2D.cpp
    src/platforms/cpu/PocketFFT3D.cpp
    src/platforms/cpu/CpuPseudoContinuous.cpp
    src/platforms/cpu/CpuPseudoDiscrete.cpp
    src/platforms/cpu/CpuAndersonMixing.cpp
    src/platforms/cpu/PocketFFTFactory.cpp
    )

#  NVIDIA CUDA
FIND_PACKAGE(CUDAToolkit)
IF(CUDAToolkit_FOUND)
    ENABLE_LANGUAGE(CUDA)
    INCLUDE_DIRECTORIES("${CUDAToolkit_INCLUDE_DIRS}")
    ADD_DEFINITIONS(-DUSE_CUDA)
    ADD_LIBRARY(cuda
        src/platforms/cuda/CudaCommon.cu
        src/platforms/cuda/CudaSimulationBox.cu
        src/platforms/cuda/CudaPseudoContinuous.cu
        src/platforms/cuda/CudaPseudoDiscrete.cu
        src/platforms/cuda/CudaCircularBuffer.cu
        src/platforms/cuda/CudaAndersonMixing.cu
        src/platforms/cuda/CudaFactory.cu
    )
    SET_PROPERTY(TARGET cuda PROPERTY CUDA_ARCHITECTURES OFF)
ENDIF()

IF( (NOT BUILD_CPU_MKL_LIB) AND
    (NOT BUILD_CPU_POCKET_FFT_LIB) AND
    (NOT CUDAToolkit_FOUND) )
    MESSAGE( FATAL_ERROR "Could not find any FFT library, CMake will exit." )
ENDIF()

#  Factory
ADD_LIBRARY(factory
    src/common/PlatformSelector.cpp
)

#---------- Python Wrapper -----------
FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
FIND_PACKAGE(pybind11 CONFIG REQUIRED)
SET(PYBIND11_CPP_STANDARD -std=c++14)

INCLUDE_DIRECTORIES(${Python3_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Python3_NumPy_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${pybind11_INCLUDE_DIRS})

PYBIND11_ADD_MODULE(langevinfts src/pybind11/langevinfts.cpp)
SET_TARGET_PROPERTIES(langevinfts PROPERTIES SUFFIX ".so")
TARGET_LINK_LIBRARIES(langevinfts PRIVATE 
    factory
	$<IF:$<BOOL:${CUDAToolkit_FOUND}>,cuda,>
	$<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cuda_driver,>
	$<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cufft,>
	$<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
	$<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-mkl,>
	$<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-qmkl,>
    $<IF:$<BOOL:${BUILD_CPU_POCKET_FFT_LIB}>,cpu-pocketfft,>
	common
)

EXECUTE_PROCESS(COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/langevinfts.so" DESTINATION ${PYTHON_SITE_PACKAGES})

#---------- Test -----------
ENABLE_TESTING()
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/tests)
