# CMake
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(LANGEVIN_FTS CXX)

SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CUDA_STANDARD 17)
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
    src/common/PropagatorComputationOptimizer.cpp
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
        src/platforms/cpu/MklFFT.cpp
        src/platforms/cpu/CpuArray.cpp
        src/platforms/cpu/CpuComputationBox.cpp
        src/platforms/cpu/CpuSolverPseudoBase.cpp
        src/platforms/cpu/CpuSolverPseudoContinuous.cpp
        src/platforms/cpu/CpuSolverPseudoDiscrete.cpp
        src/platforms/cpu/CpuSolverRealSpace.cpp
        src/platforms/cpu/CpuComputationBase.cpp
        src/platforms/cpu/CpuComputationContinuous.cpp
        src/platforms/cpu/CpuComputationDiscrete.cpp
        src/platforms/cpu/CpuComputationReduceMemoryContinuous.cpp
        src/platforms/cpu/CpuComputationReduceMemoryDiscrete.cpp
        src/platforms/cpu/CpuAndersonMixing.cpp
        src/platforms/cpu/MklFactory.cpp
    )
ELSE()
    SET(BUILD_CPU_MKL_LIB FALSE)
ENDIF()

# NVIDIA CUDA
FIND_PACKAGE(CUDAToolkit)
IF(CUDAToolkit_FOUND)
    ENABLE_LANGUAGE(CUDA)
    INCLUDE_DIRECTORIES("${CUDAToolkit_INCLUDE_DIRS}")
    ADD_DEFINITIONS(-DUSE_CUDA)
    ADD_LIBRARY(cuda
        src/platforms/cuda/CudaCommon.cu
        src/platforms/cuda/CudaArray.cu
        src/platforms/cuda/CudaComputationBox.cu
        src/platforms/cuda/CudaPseudo.cu
        src/platforms/cuda/CudaFFT.cu
        src/platforms/cuda/CudaSolverPseudoContinuous.cu
        src/platforms/cuda/CudaSolverPseudoDiscrete.cu
        src/platforms/cuda/CudaSolverRealSpace.cu
        src/platforms/cuda/CudaComputationBase.cu
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
    SET_PROPERTY(TARGET cuda PROPERTY CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")

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
SET(PYBIND11_CPP_STANDARD -std=c++17)

INCLUDE_DIRECTORIES(${Python3_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Python3_NumPy_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${pybind11_INCLUDE_DIRS})

# # Set correct module suffix for macOS (.so -> .dylib for modules, but Python expects .so)
# if(APPLE)
#     set(PYTHON_MODULE_SUFFIX ".dylib")
# else()
    set(PYTHON_MODULE_SUFFIX ".so")
# endif()

ADD_LIBRARY(_core MODULE src/pybind11/polymerfts_core.cpp) 
SET_TARGET_PROPERTIES(_core PROPERTIES PREFIX "" SUFFIX "${PYTHON_MODULE_SUFFIX}")

# Set RPATH so the installed library can find MKL at runtime
IF(BUILD_CPU_MKL_LIB AND MKL_ROOT)
    SET_TARGET_PROPERTIES(_core PROPERTIES
        INSTALL_RPATH "${MKL_ROOT}/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
    )
ENDIF()

IF(CMAKE_BUILD_TYPE STREQUAL Release)
    TARGET_COMPILE_OPTIONS(_core PRIVATE -ffast-math)
ENDIF()

TARGET_LINK_LIBRARIES(_core PUBLIC pybind11::module PRIVATE 
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

# Set install paths for Python modules
# if(APPLE)
#     execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('platlib'))" OUTPUT_VARIABLE Python3_SITEARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
#     execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" OUTPUT_VARIABLE Python3_SITELIB OUTPUT_STRIP_TRAILING_WHITESPACE)
# else()
    EXECUTE_PROCESS(COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(Python3_SITEARCH "${PYTHON_SITE_PACKAGES}")
    set(Python3_SITELIB "${PYTHON_SITE_PACKAGES}")
# endif()

# ----------------  Install rules  ----------------
# 1. compiled extension  → platform‑dependent site‑packages dir
install(TARGETS  _core
        LIBRARY  DESTINATION ${Python3_SITEARCH}/polymerfts      # Linux/macOS
        ARCHIVE  DESTINATION ${Python3_SITEARCH}/polymerfts)

# 2. pure‑Python files   → platform‑independent site‑packages dir
install(DIRECTORY  src/python/                         # contains __init__.py scfy.py lfts.py …
        DESTINATION ${Python3_SITELIB}/polymerfts)     # keeps sub‑dir structure intact

#---------- Test -----------
ENABLE_TESTING()
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/tests)
