# CMake
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

# Project version
SET(POLYMERFTS_VERSION_MAJOR 1)
SET(POLYMERFTS_VERSION_MINOR 0)
SET(POLYMERFTS_VERSION_PATCH 0)
SET(POLYMERFTS_VERSION "${POLYMERFTS_VERSION_MAJOR}.${POLYMERFTS_VERSION_MINOR}.${POLYMERFTS_VERSION_PATCH}")

PROJECT(LANGEVIN_FTS
    VERSION ${POLYMERFTS_VERSION}
    DESCRIPTION "Polymer Field Theory Simulation Library for SCFT and L-FTS"
    LANGUAGES CXX
)

# Add cmake modules path for custom finders
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build options
OPTION(POLYMERFTS_USE_FFTW "Enable FFTW3 backend" OFF)  # GPL license
OPTION(POLYMERFTS_USE_MKL "Enable Intel MKL backend" OFF)
OPTION(POLYMERFTS_USE_CUDA "Enable CUDA backend" ON)
OPTION(POLYMERFTS_BUILD_TESTS "Build test executables" ON)
OPTION(POLYMERFTS_INSTALL_PYTHON "Install Python module" ON)
OPTION(POLYMERFTS_INSTALL_HEADERS "Install C++ headers for downstream projects" OFF)

SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# NOTE: nvcc (CUDA 11.x) supports at most C++17, so headers shared between
# host and device code must stay C++17-compatible even though host-only
# code may use C++20.
SET(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# Default to an optimized build when the user does not choose one:
# an unflagged configure would otherwise build with -O0.
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    MESSAGE(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
ENDIF()

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
#---------- FTS_LIBRARIES -----------
# Basic warnings for host code in every build type (Debug adds -Wextra -pg)
ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
IF(CMAKE_BUILD_TYPE STREQUAL Debug)
    # SET(CMAKE_VERBOSE_MAKEFILE ON)
    ADD_COMPILE_OPTIONS(-Wextra -pg)
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

# spglib (required for space group symmetry)
FIND_PACKAGE(PkgConfig QUIET)
IF(PkgConfig_FOUND)
    PKG_CHECK_MODULES(SPGLIB QUIET spglib)
ENDIF()
IF(NOT SPGLIB_FOUND)
    FIND_LIBRARY(SPGLIB_LIBRARIES NAMES symspg spglib HINTS ENV SPGLIB_ROOT PATH_SUFFIXES lib lib64)
    FIND_PATH(SPGLIB_INCLUDE_DIRS spglib.h HINTS ENV SPGLIB_ROOT PATH_SUFFIXES include)
    IF(SPGLIB_LIBRARIES AND SPGLIB_INCLUDE_DIRS)
        SET(SPGLIB_FOUND TRUE)
    ENDIF()
ENDIF()
IF(SPGLIB_FOUND)
    INCLUDE_DIRECTORIES(${SPGLIB_INCLUDE_DIRS})
    MESSAGE(STATUS "spglib found: ${SPGLIB_LIBRARIES}")
ELSE()
    MESSAGE(WARNING "spglib not found. SpaceGroup functionality will not be available.")
ENDIF()

# LAPACKE (for Anderson Mixing SVD-based solver)
FIND_LIBRARY(LAPACKE_LIBRARY NAMES lapacke)
IF(LAPACKE_LIBRARY)
    MESSAGE(STATUS "LAPACKE found: ${LAPACKE_LIBRARY}")
ELSE()
    MESSAGE(WARNING "LAPACKE not found. Install with: conda install -c conda-forge lapack")
ENDIF()

# Common Files
ADD_LIBRARY(common
    src/common/Array.cpp
    src/common/ComputationBox.cpp
    src/common/PropagatorCode.cpp
    src/common/Polymer.cpp
    src/common/Molecules.cpp
    src/common/ContourLengthMapping.cpp
    src/common/PropagatorComputationOptimizer.cpp
    src/common/PropagatorAggregator.cpp
    src/common/CircularBuffer.cpp
    src/common/Pseudo.cpp
    src/common/FiniteDifference.cpp
    src/common/PropagatorComputation.cpp
    src/common/AndersonMixing.cpp
    src/common/Scheduler.cpp
    src/common/SpaceGroup.cpp
)
IF(SPGLIB_FOUND)
    TARGET_LINK_LIBRARIES(common ${SPGLIB_LIBRARIES})
ENDIF()
IF(LAPACKE_LIBRARY)
    TARGET_LINK_LIBRARIES(common ${LAPACKE_LIBRARY})
ENDIF()

# FFTW3
IF(POLYMERFTS_USE_FFTW)
    FIND_PACKAGE(PkgConfig QUIET)
    IF(PkgConfig_FOUND)
        PKG_CHECK_MODULES(FFTW3 QUIET fftw3)
    ENDIF()
    IF(NOT FFTW3_FOUND)
        FIND_LIBRARY(FFTW3_LIBRARIES NAMES fftw3 HINTS ENV FFTW_ROOT PATH_SUFFIXES lib lib64)
        FIND_PATH(FFTW3_INCLUDE_DIRS fftw3.h HINTS ENV FFTW_ROOT PATH_SUFFIXES include)
        IF(FFTW3_LIBRARIES AND FFTW3_INCLUDE_DIRS)
            SET(FFTW3_FOUND TRUE)
        ENDIF()
    ENDIF()
ENDIF()

IF(POLYMERFTS_USE_FFTW AND NOT FFTW3_FOUND)
    MESSAGE(FATAL_ERROR "POLYMERFTS_USE_FFTW=ON but FFTW3 was not found. "
        "Install fftw (e.g. conda install -c conda-forge fftw) or set FFTW_ROOT.")
ENDIF()

IF(POLYMERFTS_USE_FFTW AND FFTW3_FOUND)
    SET(BUILD_CPU_FFTW_LIB TRUE)
    ADD_DEFINITIONS(-DUSE_CPU_FFTW)
    INCLUDE_DIRECTORIES(${FFTW3_INCLUDE_DIRS})
    # Put the FFTW library directory on the linker search path so short-form
    # (-lfftw3) references resolve without relying on LIBRARY_PATH in the shell.
    IF(FFTW3_LIBRARY_DIRS)
        LINK_DIRECTORIES(${FFTW3_LIBRARY_DIRS})
    ENDIF()
    FOREACH(_fftw_lib ${FFTW3_LIBRARIES})
        IF(EXISTS "${_fftw_lib}")
            GET_FILENAME_COMPONENT(_fftw_dir "${_fftw_lib}" DIRECTORY)
            LINK_DIRECTORIES(${_fftw_dir})
        ENDIF()
    ENDFOREACH()
    ADD_LIBRARY(cpu-fftw
        src/platforms/cpu/FftwFFT.cpp
        src/platforms/cpu/FftwCrysFFTPmmm.cpp
        src/platforms/cpu/FftwCrysFFTRecursive3m.cpp
        src/platforms/cpu/FftwCrysFFTObliqueZ.cpp
        src/platforms/cpu/CpuArray.cpp
        src/platforms/cpu/CpuComputationBox.cpp
        src/platforms/cpu/CpuSolverPseudoBase.cpp
        src/platforms/cpu/CpuSolverPseudoRQM4.cpp
        src/platforms/cpu/CpuSolverPseudoRK2.cpp
        src/platforms/cpu/CpuSolverPseudoDiscrete.cpp
        src/platforms/cpu/CpuSolverCNADI.cpp
        src/platforms/cpu/CpuComputationBase.cpp
        src/platforms/cpu/CpuComputationContinuous.cpp
        src/platforms/cpu/CpuComputationDiscrete.cpp
        src/platforms/cpu/CpuComputationReduceMemoryBase.cpp
        src/platforms/cpu/CpuComputationReduceMemoryContinuous.cpp
        src/platforms/cpu/CpuComputationReduceMemoryDiscrete.cpp
        src/platforms/cpu/CpuAndersonMixing.cpp
        src/platforms/cpu/FftwFactory.cpp
    )
    TARGET_COMPILE_DEFINITIONS(cpu-fftw PRIVATE USE_CPU_FFTW)
    # PUBLIC so consumers (tests, _core) inherit FFTW transitively and no
    # raw -lfftw3 flags are needed anywhere.
    TARGET_LINK_LIBRARIES(cpu-fftw PUBLIC ${FFTW3_LIBRARIES})

    # GPL License Warning
    MESSAGE(WARNING
        "================================================================================\n"
        "  GPL LICENSE WARNING\n"
        "================================================================================\n"
        "  FFTW3 is licensed under the GNU General Public License (GPL).\n"
        "\n"
        "  If you distribute binaries compiled with POLYMERFTS_USE_FFTW=ON,\n"
        "  you MUST comply with the GPL license terms, which include:\n"
        "\n"
        "    1. Distribute the complete source code of your application\n"
        "    2. License your application under GPL or a GPL-compatible license\n"
        "    3. Include the full GPL license text with your distribution\n"
        "\n"
        "  For internal/personal use only, no action is required.\n"
        "  For more information, see: https://www.fftw.org/faq/section1.html\n"
        "================================================================================\n"
    )
ELSE()
    SET(BUILD_CPU_FFTW_LIB FALSE)
ENDIF()

# Intel MKL
IF(POLYMERFTS_USE_MKL)
    FIND_PACKAGE(MKL QUIET)
ENDIF()

IF(POLYMERFTS_USE_MKL AND NOT MKL_FOUND)
    MESSAGE(FATAL_ERROR "POLYMERFTS_USE_MKL=ON but MKL was not found. "
        "Install mkl-devel (e.g. conda install -c conda-forge mkl mkl-devel mkl-include) "
        "or set MKLROOT.")
ENDIF()

IF(POLYMERFTS_USE_MKL AND MKL_FOUND)
    SET(BUILD_CPU_MKL_LIB TRUE)
    ADD_DEFINITIONS(-DUSE_CPU_MKL)
    INCLUDE_DIRECTORIES(${MKL_INCLUDE_DIRS})
    ADD_LIBRARY(cpu-mkl
        src/platforms/cpu/MklFFT.cpp
        src/platforms/cpu/MklRealTransform.cpp
        src/platforms/cpu/MklCrysFFTPmmm.cpp
        src/platforms/cpu/MklCrysFFTRecursive3m.cpp
        src/platforms/cpu/MklCrysFFTObliqueZ.cpp
        src/platforms/cpu/CpuArray.cpp
        src/platforms/cpu/CpuComputationBox.cpp
        src/platforms/cpu/CpuSolverPseudoBase.cpp
        src/platforms/cpu/CpuSolverPseudoRQM4.cpp
        src/platforms/cpu/CpuSolverPseudoRK2.cpp
        src/platforms/cpu/CpuSolverPseudoDiscrete.cpp
        src/platforms/cpu/CpuSolverCNADI.cpp
        src/platforms/cpu/CpuComputationBase.cpp
        src/platforms/cpu/CpuComputationContinuous.cpp
        src/platforms/cpu/CpuComputationDiscrete.cpp
        src/platforms/cpu/CpuComputationReduceMemoryBase.cpp
        src/platforms/cpu/CpuComputationReduceMemoryContinuous.cpp
        src/platforms/cpu/CpuComputationReduceMemoryDiscrete.cpp
        src/platforms/cpu/CpuAndersonMixing.cpp
        src/platforms/cpu/MklFactory.cpp
    )
    TARGET_COMPILE_DEFINITIONS(cpu-mkl PRIVATE USE_CPU_MKL)
    TARGET_LINK_LIBRARIES(cpu-mkl ${MKL_LIBRARIES})
ELSE()
    SET(BUILD_CPU_MKL_LIB FALSE)
ENDIF()

# NVIDIA CUDA
IF(POLYMERFTS_USE_CUDA)
    FIND_PACKAGE(CUDAToolkit QUIET)
ENDIF()

IF(POLYMERFTS_USE_CUDA AND NOT CUDAToolkit_FOUND)
    # CUDA defaults to ON, so a missing toolkit is only a warning: CPU-only
    # machines should still configure. Explicitly disable with
    # -DPOLYMERFTS_USE_CUDA=OFF to silence this.
    MESSAGE(WARNING "POLYMERFTS_USE_CUDA=ON but no CUDA toolkit was found; "
        "building without the GPU backend.")
ENDIF()

IF(POLYMERFTS_USE_CUDA AND CUDAToolkit_FOUND)
    ENABLE_LANGUAGE(CUDA)
    INCLUDE_DIRECTORIES("${CUDAToolkit_INCLUDE_DIRS}")
    # Put the CUDA toolkit library directory on the linker search path so the static
    # CUDA runtime references (-lcudart_static, -lcudadevrt) resolve without relying
    # on LIBRARY_PATH being set in the shell (e.g. by `module load cuda`).
    IF(CUDAToolkit_LIBRARY_DIR)
        LINK_DIRECTORIES(${CUDAToolkit_LIBRARY_DIR})
    ENDIF()
    ADD_DEFINITIONS(-DUSE_CUDA)
    ADD_LIBRARY(cuda
        src/platforms/cuda/CudaCommon.cu
        src/platforms/cuda/CudaArray.cu
        src/platforms/cuda/CudaComputationBox.cu
        src/platforms/cuda/CudaPseudo.cu
        src/platforms/cuda/CudaFFT.cu
        src/platforms/cuda/CudaCrysFFT.cu
        src/platforms/cuda/CudaCrysFFTRecursive3m.cu
        src/platforms/cuda/CudaCrysFFTObliqueZ.cu
        src/platforms/cuda/CudaCrysFFTStress.cu
        src/platforms/cuda/CudaRealTransform.cu
        src/platforms/cuda/CudaSolverPseudoRQM4.cu
        src/platforms/cuda/CudaSolverPseudoRK2.cu
        src/platforms/cuda/CudaSolverPseudoDiscrete.cu
        src/platforms/cuda/CudaSolverCNADI.cu
        src/platforms/cuda/CudaComputationBase.cu
        src/platforms/cuda/CudaComputationContinuous.cu
        src/platforms/cuda/CudaComputationDiscrete.cu
        src/platforms/cuda/CudaComputationReduceMemoryBase.cu
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

IF( (NOT BUILD_CPU_FFTW_LIB) AND
    (NOT BUILD_CPU_MKL_LIB) AND
    (NOT CUDAToolkit_FOUND) )
    MESSAGE( FATAL_ERROR "Could not find any FFT library (FFTW, MKL, or CUDA), CMake will exit." )
ENDIF()

#  Factory
ADD_LIBRARY(factory
    src/common/PlatformSelector.cpp
)

#---------- Python Wrapper -----------
FIND_PACKAGE(Python3 3.11 COMPONENTS Interpreter Development NumPy REQUIRED)
FIND_PACKAGE(pybind11 2.13 REQUIRED)
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

IF(CMAKE_BUILD_TYPE STREQUAL Release)
ENDIF()

TARGET_LINK_LIBRARIES(_core PUBLIC pybind11::module PRIVATE
    factory
    $<IF:$<BOOL:${CUDAToolkit_FOUND}>,cuda,>
    $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cufft,>
    $<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,cpu-fftw,>
    $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
    common
)

# Set install paths for Python modules.
IF(SKBUILD)
    # Building a wheel via scikit-build-core (pip install .): relative
    # destinations are mapped into the wheel's platlib automatically.
    SET(POLYMERFTS_PY_INSTALL_DIR "polymerfts")
ELSE()
    # Direct 'make install': query the SAME interpreter found by
    # FIND_PACKAGE(Python3) — a bare "python" may be a different (system)
    # interpreter and would install into the wrong site-packages.
    # sysconfig replaces distutils, which was removed in Python 3.12.
    EXECUTE_PROCESS(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'))"
        OUTPUT_VARIABLE Python3_SITEARCH
        RESULT_VARIABLE _sitearch_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF(NOT _sitearch_result EQUAL 0)
        MESSAGE(FATAL_ERROR "Failed to query site-packages path from ${Python3_EXECUTABLE}")
    ENDIF()
    SET(POLYMERFTS_PY_INSTALL_DIR "${Python3_SITEARCH}/polymerfts")
ENDIF()

# ----------------  Install rules  ----------------
IF(POLYMERFTS_INSTALL_PYTHON)
    # 1. compiled extension
    install(TARGETS  _core
            LIBRARY  DESTINATION ${POLYMERFTS_PY_INSTALL_DIR}      # Linux/macOS
            ARCHIVE  DESTINATION ${POLYMERFTS_PY_INSTALL_DIR}
            COMPONENT python)

    # 2. pure‑Python files
    install(DIRECTORY  src/python/                         # contains __init__.py scft.py lfts.py …
            DESTINATION ${POLYMERFTS_PY_INSTALL_DIR}       # keeps sub‑dir structure intact
            COMPONENT python
            FILES_MATCHING PATTERN "*.py")
ENDIF()

# Install headers for downstream C++ projects
IF(POLYMERFTS_INSTALL_HEADERS)
    install(DIRECTORY src/common/
            DESTINATION include/polymerfts
            COMPONENT development
            FILES_MATCHING PATTERN "*.h")
ENDIF()

#---------- Test -----------
IF(POLYMERFTS_BUILD_TESTS)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/tests)
ENDIF()

#---------- CMake Config Export -----------
IF(POLYMERFTS_INSTALL_HEADERS)
    INCLUDE(CMakePackageConfigHelpers)

    # Generate version file
    WRITE_BASIC_PACKAGE_VERSION_FILE(
        "${CMAKE_CURRENT_BINARY_DIR}/polymerfts-config-version.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    # Generate config file
    CONFIGURE_PACKAGE_CONFIG_FILE(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/polymerfts-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/polymerfts-config.cmake"
        INSTALL_DESTINATION lib/cmake/polymerfts
    )

    # Install CMake config files
    INSTALL(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/polymerfts-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/polymerfts-config-version.cmake"
        DESTINATION lib/cmake/polymerfts
        COMPONENT development
    )
ENDIF()
