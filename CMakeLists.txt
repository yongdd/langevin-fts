# CMake
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(LANGEVIN_FTS CXX)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CUDA_STANDARD 14)
SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# SET(CMAKE_BUILD_TYPE Release)
# SET(CMAKE_BUILD_TYPE Debug  )

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
#---------- FTS_LIBRARIES -----------
# IF(CMAKE_BUILD_TYPE STREQUAL Release)
#     ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
# ENDIF()
IF(CMAKE_BUILD_TYPE STREQUAL Debug)
    # SET(CMAKE_VERBOSE_MAKEFILE ON)
    ADD_COMPILE_OPTIONS(-Wall -Wextra -pg)
    # -fsanitize=address -fno-omit-frame-pointer)
    # SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" -fsanitize=address -fno-omit-frame-pointer)
    # SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" -fsanitize=address -fno-omit-frame-pointer)
ENDIF()

# OpenMP
FIND_PACKAGE(OpenMP REQUIRED)
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
    src/common/Array.cpp
    src/common/ComputationBox.cpp
    src/common/PropagatorCode.cpp
    src/common/Polymer.cpp
    src/common/Molecules.cpp
    src/common/PropagatorAnalyzer.cpp
    src/common/CircularBuffer.cpp
    src/common/Pseudo.cpp
    src/common/FiniteDifference.cpp
    src/common/PropagatorComputation.cpp
    src/common/AndersonMixing.cpp
    src/common/Scheduler.cpp
)

# Intel MKL
SET(ENV{MKL_ROOT} "$ENV{MKLROOT}")
FIND_PACKAGE(MKL)
IF(MKL_FOUND)
    SET(BUILD_CPU_MKL_LIB TRUE)
    ADD_DEFINITIONS(-DUSE_CPU_MKL)
    ADD_LIBRARY(cpu-mkl
        src/platforms/cpu/MklFFT1D.cpp
        src/platforms/cpu/MklFFT2D.cpp
        src/platforms/cpu/MklFFT3D.cpp
        src/platforms/cpu/CpuArray.cpp
        src/platforms/cpu/CpuComputationBox.cpp
        src/platforms/cpu/CpuSolverPseudo.cpp
        src/platforms/cpu/CpuSolverReal.cpp
        src/platforms/cpu/CpuComputationContinuous.cpp
        src/platforms/cpu/CpuComputationDiscrete.cpp
        src/platforms/cpu/CpuAndersonMixing.cpp
        src/platforms/cpu/MklFactory.cpp
    )
ELSE()
    SET(BUILD_CPU_MKL_LIB FALSE)
ENDIF()

#  NVIDIA CUDA
FIND_PACKAGE(CUDAToolkit)
IF(CUDAToolkit_FOUND)
    ENABLE_LANGUAGE(CUDA)
    INCLUDE_DIRECTORIES("${CUDAToolkit_INCLUDE_DIRS}")
    ADD_DEFINITIONS(-DUSE_CUDA)
    ADD_LIBRARY(cuda
        src/platforms/cuda/CudaCommon.cu
        src/platforms/cuda/CudaArray.cu
        src/platforms/cuda/CudaComputationBox.cu
        src/platforms/cuda/CudaSolverPseudo.cu
        src/platforms/cuda/CudaSolverReal.cu
        src/platforms/cuda/CudaComputationContinuous.cu
        src/platforms/cuda/CudaComputationDiscrete.cu
        src/platforms/cuda/CudaComputationReduceMemoryContinuous.cu
        src/platforms/cuda/CudaComputationReduceMemoryDiscrete.cu
        src/platforms/cuda/CudaCircularBuffer.cu
        src/platforms/cuda/PinnedCircularBuffer.cu
        src/platforms/cuda/CudaAndersonMixing.cu
        src/platforms/cuda/CudaAndersonMixingReduceMemory.cu
        src/platforms/cuda/CudaFactory.cu
    )
    SET_PROPERTY(TARGET cuda PROPERTY CUDA_ARCHITECTURES OFF)

    IF (OPENMP_FOUND)
        TARGET_COMPILE_OPTIONS(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
    ENDIF()

ENDIF()

IF( (NOT BUILD_CPU_MKL_LIB) AND
    (NOT CUDAToolkit_FOUND) )
    MESSAGE( FATAL_ERROR "Could not find any FFT library, CMake will exit." )
ENDIF()

#  Factory
ADD_LIBRARY(factory
    src/common/PlatformSelector.cpp
)

#---------- Python Wrapper -----------
FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
FIND_PACKAGE(pybind11 REQUIRED)
SET(PYBIND11_CPP_STANDARD -std=c++14)

INCLUDE_DIRECTORIES(${Python3_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Python3_NumPy_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${pybind11_INCLUDE_DIRS})

# PYBIND11_ADD_MODULE(langevinfts src/pybind11/langevinfts.cpp) // new find python mode can cause segfault error depending on compile environment
ADD_LIBRARY(langevinfts MODULE src/pybind11/langevinfts.cpp) 
SET_TARGET_PROPERTIES(langevinfts PROPERTIES PREFIX "" SUFFIX ".so")

IF(CMAKE_BUILD_TYPE STREQUAL Release)
    TARGET_COMPILE_OPTIONS(langevinfts PRIVATE -ffast-math)
ENDIF()

TARGET_LINK_LIBRARIES(langevinfts PUBLIC pybind11::module PRIVATE 
    factory
    $<IF:$<BOOL:${CUDAToolkit_FOUND}>,cuda,>
    $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cufft,>
    $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
    $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lmkl_intel_lp64,>
    $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lmkl_sequential,>
    $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lmkl_core,>
    $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-ldl,>
    $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lpthread,>
    $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lm,>
    common
)

EXECUTE_PROCESS(COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/langevinfts.so" DESTINATION ${PYTHON_SITE_PACKAGES})

#---------- Test -----------
ENABLE_TESTING()
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/tests)
