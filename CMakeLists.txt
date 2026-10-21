CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

# ============================================================================
#  Project
# ============================================================================
SET(POLYMERFTS_VERSION_MAJOR 0)
SET(POLYMERFTS_VERSION_MINOR 10)
SET(POLYMERFTS_VERSION_PATCH 0)
SET(POLYMERFTS_VERSION "${POLYMERFTS_VERSION_MAJOR}.${POLYMERFTS_VERSION_MINOR}.${POLYMERFTS_VERSION_PATCH}")

PROJECT(LANGEVIN_FTS
    VERSION ${POLYMERFTS_VERSION}
    DESCRIPTION "Polymer Field Theory Simulation Library for SCFT and L-FTS"
    LANGUAGES CXX
)

# Custom find modules (FindMKL.cmake)
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ============================================================================
#  Options
# ============================================================================
OPTION(POLYMERFTS_USE_FFTW        "Enable FFTW3 backend (GPL license)"        OFF)
OPTION(POLYMERFTS_USE_MKL         "Enable Intel MKL backend"                  OFF)
OPTION(POLYMERFTS_USE_CUDA        "Enable CUDA backend"                       ON)
OPTION(POLYMERFTS_BUILD_TESTS     "Build test executables"                    ON)
OPTION(POLYMERFTS_INSTALL_PYTHON  "Install Python module"                     ON)
OPTION(POLYMERFTS_INSTALL_HEADERS "Install C++ headers for downstream projects" OFF)
OPTION(POLYMERFTS_WERROR          "Treat host C++ warnings as errors (CI)"    OFF)
SET(POLYMERFTS_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90" CACHE STRING
    "CUDA compute capabilities to compile for (semicolon-separated)")

# ============================================================================
#  Compiler settings
# ============================================================================
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# Device code is C++20 too (nvcc 12.2+; std::source_location needs 12.2, see the
# CUDA backend section below).
SET(CMAKE_CUDA_STANDARD 20)
SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# Default to an optimized build when the user does not choose one:
# an unflagged configure would otherwise build with -O0.
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    MESSAGE(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
ENDIF()

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The installed _core.so records the directories of the libraries it was linked
# against (conda lib, CUDA lib), so importing it does not need LD_LIBRARY_PATH.
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Warnings for host code in every build type; Debug adds -Wextra and gprof
# instrumentation. (For AddressSanitizer add -fsanitize=address
# -fno-omit-frame-pointer to CMAKE_CXX_FLAGS on the command line.)
ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
IF(POLYMERFTS_WERROR)
    ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
ENDIF()
IF(CMAKE_BUILD_TYPE STREQUAL Debug)
    # host code only: nvcc rejects -Wextra
    ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-pg>)
    ADD_LINK_OPTIONS(-pg)
ENDIF()

# OpenMP (propagator-level parallelism on the CPU; the CUDA target adds
# -fopenmp for its host code below)
FIND_PACKAGE(OpenMP REQUIRED)
SET(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/common/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/platforms/cpu/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/platforms/cuda/)

# ============================================================================
#  Required dependencies: spglib, LAPACKE
# ============================================================================
# spglib: space group symmetry. SpaceGroup.cpp is always compiled and includes
# spglib.h, so a missing spglib must stop configuration here rather than
# failing mid-compile.
# Libraries are always linked by absolute path (pkg-config's *_LINK_LIBRARIES,
# find_library results): the generated link commands must not depend on
# LIBRARY_PATH or -L directories that happen to be set in the developer's shell.
FIND_PACKAGE(PkgConfig QUIET)
IF(PkgConfig_FOUND)
    PKG_CHECK_MODULES(SPGLIB QUIET spglib)
    IF(SPGLIB_FOUND)
        SET(SPGLIB_LIBRARIES "${SPGLIB_LINK_LIBRARIES}")
    ENDIF()
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
    MESSAGE(FATAL_ERROR "spglib is required but was not found. "
        "Install with: conda install -c conda-forge spglib "
        "(or apt install libsymspg-dev), or set SPGLIB_ROOT.")
ENDIF()

# LAPACKE: SVD solver of Anderson mixing. AndersonMixing.cpp includes lapacke.h or,
# failing that, MKL's mkl_lapacke.h; the library is linked into 'common', so the
# library and one of the two headers are required at configure time. (conda-forge's
# liblapacke built against MKL ships only the shared library; its header is
# mkl_lapacke.h from mkl-include.)
FIND_LIBRARY(LAPACKE_LIBRARY NAMES lapacke HINTS ENV LAPACKE_ROOT PATH_SUFFIXES lib lib64)
FIND_PATH(LAPACKE_INCLUDE_DIR NAMES lapacke.h mkl_lapacke.h HINTS ENV LAPACKE_ROOT ENV MKLROOT PATH_SUFFIXES include)
IF(LAPACKE_LIBRARY AND LAPACKE_INCLUDE_DIR)
    INCLUDE_DIRECTORIES(${LAPACKE_INCLUDE_DIR})
    MESSAGE(STATUS "LAPACKE found: ${LAPACKE_LIBRARY} (${LAPACKE_INCLUDE_DIR})")
ELSE()
    MESSAGE(FATAL_ERROR "LAPACKE (library and lapacke.h or mkl_lapacke.h) is required but was not found. "
        "Install with: conda install -c conda-forge liblapacke (plus mkl-include for the MKL build), "
        "or apt install liblapacke-dev, or set LAPACKE_ROOT.")
ENDIF()

# ============================================================================
#  Platform-independent library
# ============================================================================
ADD_LIBRARY(common
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
    src/common/ComputationCommon.cpp
    src/common/ComputationContinuous.cpp
    src/common/ComputationDiscrete.cpp
    src/common/ComputationReduceMemoryBase.cpp
    src/common/ComputationReduceMemoryContinuous.cpp
    src/common/ComputationReduceMemoryDiscrete.cpp
    src/common/AndersonMixing.cpp
    src/common/Scheduler.cpp
    src/common/SpaceGroup.cpp
)
TARGET_LINK_LIBRARIES(common ${SPGLIB_LIBRARIES} ${LAPACKE_LIBRARY})

# ============================================================================
#  CPU backends (FFTW and/or MKL)
# ============================================================================
# Sources shared by both CPU backends. USE_CPU_FFTW / USE_CPU_MKL are global
# definitions, so these objects are the same for both backends (the transform
# backend is chosen at run time via FFTBackend); compile them once.
SET(POLYMERFTS_CPU_COMMON_SOURCES
    src/platforms/cpu/CpuComputationBox.cpp
    src/platforms/cpu/CpuSolverPseudoBase.cpp
    src/platforms/cpu/CpuSolverPseudoRQM4.cpp
    src/platforms/cpu/CpuSolverPseudoRK2.cpp
    src/platforms/cpu/CpuSolverPseudoDiscrete.cpp
    src/platforms/cpu/CpuSolverCNADI.cpp
    src/platforms/cpu/CpuComputationBackend.cpp
    src/platforms/cpu/CpuAndersonMixing.cpp
)

# ---- FFTW3 ----
SET(BUILD_CPU_FFTW_LIB FALSE)
IF(POLYMERFTS_USE_FFTW)
    IF(PkgConfig_FOUND)
        PKG_CHECK_MODULES(FFTW3 QUIET fftw3)
        IF(FFTW3_FOUND)
            SET(FFTW3_LIBRARIES "${FFTW3_LINK_LIBRARIES}")
        ENDIF()
    ENDIF()
    IF(NOT FFTW3_FOUND)
        FIND_LIBRARY(FFTW3_LIBRARIES NAMES fftw3 HINTS ENV FFTW_ROOT PATH_SUFFIXES lib lib64)
        FIND_PATH(FFTW3_INCLUDE_DIRS fftw3.h HINTS ENV FFTW_ROOT PATH_SUFFIXES include)
        IF(FFTW3_LIBRARIES AND FFTW3_INCLUDE_DIRS)
            SET(FFTW3_FOUND TRUE)
        ENDIF()
    ENDIF()
    IF(NOT FFTW3_FOUND)
        MESSAGE(FATAL_ERROR "POLYMERFTS_USE_FFTW=ON but FFTW3 was not found. "
            "Install fftw (e.g. conda install -c conda-forge fftw) or set FFTW_ROOT.")
    ENDIF()

    SET(BUILD_CPU_FFTW_LIB TRUE)
    ADD_DEFINITIONS(-DUSE_CPU_FFTW)
    INCLUDE_DIRECTORIES(${FFTW3_INCLUDE_DIRS})

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
ENDIF()

# ---- Intel MKL ----
SET(BUILD_CPU_MKL_LIB FALSE)
IF(POLYMERFTS_USE_MKL)
    FIND_PACKAGE(MKL QUIET)
    IF(NOT MKL_FOUND)
        MESSAGE(FATAL_ERROR "POLYMERFTS_USE_MKL=ON but MKL was not found. "
            "Install MKL (e.g. conda install -c conda-forge mkl mkl-include) "
            "or set MKLROOT.")
    ENDIF()

    SET(BUILD_CPU_MKL_LIB TRUE)
    ADD_DEFINITIONS(-DUSE_CPU_MKL)
    INCLUDE_DIRECTORIES(${MKL_INCLUDE_DIRS})
ENDIF()

# ---- CPU backend libraries ----
IF(BUILD_CPU_FFTW_LIB OR BUILD_CPU_MKL_LIB)
    ADD_LIBRARY(cpu-common OBJECT ${POLYMERFTS_CPU_COMMON_SOURCES})
ENDIF()
IF(BUILD_CPU_FFTW_LIB)
    ADD_LIBRARY(cpu-fftw
        src/platforms/cpu/FftwFFT.cpp
        src/platforms/cpu/FftwCrysFFTPmmm.cpp
        src/platforms/cpu/FftwCrysFFTRecursive3m.cpp
        src/platforms/cpu/FftwCrysFFTObliqueZ.cpp
        src/platforms/cpu/FftwFactory.cpp
        $<TARGET_OBJECTS:cpu-common>
    )
    # PUBLIC so consumers (tests, _core) inherit FFTW transitively and no
    # raw -lfftw3 flags are needed anywhere.
    TARGET_LINK_LIBRARIES(cpu-fftw PUBLIC ${FFTW3_LIBRARIES})
ENDIF()
IF(BUILD_CPU_MKL_LIB)
    ADD_LIBRARY(cpu-mkl
        src/platforms/cpu/MklFFT.cpp
        src/platforms/cpu/MklRealTransform.cpp
        src/platforms/cpu/MklCrysFFTPmmm.cpp
        src/platforms/cpu/MklCrysFFTRecursive3m.cpp
        src/platforms/cpu/MklCrysFFTObliqueZ.cpp
        src/platforms/cpu/MklFactory.cpp
        $<TARGET_OBJECTS:cpu-common>
    )
    TARGET_LINK_LIBRARIES(cpu-mkl ${MKL_LIBRARIES})
ENDIF()

# ============================================================================
#  CUDA backend
# ============================================================================
IF(POLYMERFTS_USE_CUDA)
    FIND_PACKAGE(CUDAToolkit QUIET)
    IF(NOT CUDAToolkit_FOUND)
        # CUDA defaults to ON, so a missing toolkit is only a warning: CPU-only
        # machines should still configure. Use -DPOLYMERFTS_USE_CUDA=OFF to silence it.
        MESSAGE(WARNING "POLYMERFTS_USE_CUDA=ON but no CUDA toolkit was found; "
            "building without the GPU backend.")
    ENDIF()
ENDIF()

IF(POLYMERFTS_USE_CUDA AND CUDAToolkit_FOUND)
    # The .cu files are C++20: nvcc 11.x has no -std=c++20, and nvcc 12.0/12.1
    # cannot evaluate std::source_location::current() (used by CudaBuffer to name
    # the caller of a failed allocation).
    IF(CUDAToolkit_VERSION VERSION_LESS 12.2)
        MESSAGE(FATAL_ERROR "CUDA toolkit ${CUDAToolkit_VERSION} found at ${CUDAToolkit_TARGET_DIR}; "
            "the GPU backend needs CUDA 12.2 or newer (C++20 device code). Configure a new build "
            "directory (or delete CMakeCache.txt) with a newer toolkit's nvcc first on PATH or "
            "given as -DCMAKE_CUDA_COMPILER=..., or configure with -DPOLYMERFTS_USE_CUDA=OFF.")
    ENDIF()
    # CMake passes -std=c++20 to nvcc from 3.25.2 on
    IF(CMAKE_VERSION VERSION_LESS 3.25.2)
        MESSAGE(FATAL_ERROR "CMake ${CMAKE_VERSION}: the GPU backend needs CMake 3.25.2 or newer "
            "(C++20 device code; environment.yml installs 3.31), or configure with "
            "-DPOLYMERFTS_USE_CUDA=OFF.")
    ENDIF()
    # CudaCommon.h, which host C++ sources include as well, uses std::source_location
    # (libstdc++ of GCC 11+, Clang 15+)
    INCLUDE(CheckCXXSourceCompiles)
    CHECK_CXX_SOURCE_COMPILES("
        #include <source_location>
        int line(std::source_location where = std::source_location::current())
        { return static_cast<int>(where.line()); }
        int main() { return line() > 0 ? 0 : 1; }" POLYMERFTS_HAVE_SOURCE_LOCATION)
    IF(NOT POLYMERFTS_HAVE_SOURCE_LOCATION)
        MESSAGE(FATAL_ERROR "The C++ compiler (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}) "
            "has no std::source_location, which the GPU backend uses: use GCC 11+ or Clang 15+, "
            "or configure with -DPOLYMERFTS_USE_CUDA=OFF.")
    ENDIF()
    ENABLE_LANGUAGE(CUDA)
    MESSAGE(STATUS "CUDA toolkit ${CUDAToolkit_VERSION}: ${CUDAToolkit_TARGET_DIR}")

    # A developer's shell often exports the toolkit directories in
    # CPLUS_INCLUDE_PATH and LIBRARY_PATH (e.g. `module load cuda`). CMake then
    # records them as compiler-implicit directories and omits the -I flag for
    # cufft.h in host code and the -L flag the CUDA runtime (-lcudart_static,
    # -lcudadevrt) needs, so the generated build only works inside that shell.
    # Remove the toolkit directories from the implicit lists so the flags are
    # always emitted and the build tree is self-contained.
    GET_FILENAME_COMPONENT(_cuda_root "${CUDAToolkit_TARGET_DIR}" REALPATH)
    FOREACH(_dir ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
        GET_FILENAME_COMPONENT(_real "${_dir}" REALPATH)
        STRING(FIND "${_real}/" "${_cuda_root}/" _pos)   # whole path components only
        IF(_pos EQUAL 0)
            LIST(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "${_dir}")
        ENDIF()
    ENDFOREACH()
    FOREACH(_dir ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
        GET_FILENAME_COMPONENT(_real "${_dir}" REALPATH)
        STRING(FIND "${_real}/" "${_cuda_root}/" _pos)
        IF(_pos EQUAL 0)
            LIST(REMOVE_ITEM CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "${_dir}")
        ENDIF()
    ENDFOREACH()

    # Link the cuFFT of the toolkit whose nvcc and cufft.h build the code.
    # FindCUDAToolkit looks for the libraries on CMAKE_PREFIX_PATH first, so a
    # conda environment that carries another CUDA's runtime libraries (e.g.
    # cudatoolkit for CuPy) would otherwise supply a cuFFT of another version.
    FOREACH(_dir "${CUDAToolkit_TARGET_DIR}/lib64" "${CUDAToolkit_TARGET_DIR}/lib")
        SET(_toolkit_cufft "${_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}cufft${CMAKE_SHARED_LIBRARY_SUFFIX}")
        IF(EXISTS "${_toolkit_cufft}")
            SET_TARGET_PROPERTIES(CUDA::cufft PROPERTIES IMPORTED_LOCATION "${_toolkit_cufft}")
            BREAK()
        ENDIF()
    ENDFOREACH()
    GET_TARGET_PROPERTY(_cufft_location CUDA::cufft IMPORTED_LOCATION)
    MESSAGE(STATUS "cuFFT: ${_cufft_location}")

    # Host-compiled sources (PlatformSelector.cpp, the pybind module, tests)
    # include cufft.h, so the toolkit headers go on the C++ include path too.
    INCLUDE_DIRECTORIES("${CUDAToolkit_INCLUDE_DIRS}")
    ADD_DEFINITIONS(-DUSE_CUDA)
    ADD_LIBRARY(cuda
        src/platforms/cuda/CudaCommon.cu
        src/platforms/cuda/CudaComputationBox.cu
        src/platforms/cuda/CudaPseudo.cu
        src/platforms/cuda/CudaFFT.cu
        src/platforms/cuda/CudaCrysFFT.cu
        src/platforms/cuda/CudaCrysFFTRecursive3m.cu
        src/platforms/cuda/CudaCrysFFTObliqueZ.cu
        src/platforms/cuda/CudaCrysFFTStress.cu
        src/platforms/cuda/CudaRealTransform.cu
        src/platforms/cuda/CudaSolverPseudoBase.cu
        src/platforms/cuda/CudaSolverPseudoRQM4.cu
        src/platforms/cuda/CudaSolverPseudoRK2.cu
        src/platforms/cuda/CudaSolverPseudoDiscrete.cu
        src/platforms/cuda/CudaSolverCNADI.cu
        src/platforms/cuda/CudaComputationBackend.cu
        src/platforms/cuda/CudaCircularBuffer.cu
        src/platforms/cuda/PinnedCircularBuffer.cu
        src/platforms/cuda/CudaAndersonMixing.cu
        src/platforms/cuda/CudaAndersonMixingReduceMemory.cu
        src/platforms/cuda/CudaFactory.cu
    )
    SET_PROPERTY(TARGET cuda PROPERTY CUDA_ARCHITECTURES ${POLYMERFTS_CUDA_ARCHITECTURES})
    TARGET_COMPILE_OPTIONS(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
ENDIF()

IF(NOT BUILD_CPU_FFTW_LIB AND NOT BUILD_CPU_MKL_LIB AND NOT CUDAToolkit_FOUND)
    MESSAGE(FATAL_ERROR "No computational backend enabled: turn on at least one of "
        "POLYMERFTS_USE_MKL, POLYMERFTS_USE_FFTW, or POLYMERFTS_USE_CUDA (with a CUDA toolkit).")
ENDIF()

# Platform selector: depends on every enabled backend, hence a separate target
ADD_LIBRARY(factory
    src/common/PlatformSelector.cpp
)

# Everything a consumer of the library links (the Python module, tests,
# benchmarks): selector, enabled backends, common core. Order matters for
# static libraries.
SET(POLYMERFTS_BACKEND_LIBS factory)
IF(CUDAToolkit_FOUND)
    LIST(APPEND POLYMERFTS_BACKEND_LIBS cuda CUDA::cufft)
ENDIF()
IF(BUILD_CPU_FFTW_LIB)
    LIST(APPEND POLYMERFTS_BACKEND_LIBS cpu-fftw)
ENDIF()
IF(BUILD_CPU_MKL_LIB)
    LIST(APPEND POLYMERFTS_BACKEND_LIBS cpu-mkl)
ENDIF()
LIST(APPEND POLYMERFTS_BACKEND_LIBS common)
ADD_LIBRARY(polymerfts_backends INTERFACE)
TARGET_LINK_LIBRARIES(polymerfts_backends INTERFACE ${POLYMERFTS_BACKEND_LIBS})

# ============================================================================
#  Python module (pybind11)
# ============================================================================
# Development.Module (not Development): an extension module needs the headers
# but no libpython to embed, which some environments (manylinux, venvs) lack.
# pybind11/numpy.h talks to NumPy through its Python API, so no NumPy headers.
FIND_PACKAGE(Python3 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
# pybind11 from conda (CMAKE_PREFIX_PATH) or from pip (its cmake dir is asked
# from the interpreter)
EXECUTE_PROCESS(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
FIND_PACKAGE(pybind11 2.13 REQUIRED HINTS ${_pybind11_cmakedir})

# pybind11::module carries the Python and pybind11 include directories
ADD_LIBRARY(_core MODULE src/pybind11/polymerfts_core.cpp)
SET_TARGET_PROPERTIES(_core PROPERTIES PREFIX "" SUFFIX ".so")
TARGET_LINK_LIBRARIES(_core PUBLIC pybind11::module PRIVATE polymerfts_backends)

# Stage an importable 'polymerfts' package inside the build tree on every
# build. The Python tests put this directory on PYTHONPATH so they always
# exercise the code that was just built, never a stale site-packages install.
# (cmake/stage_python.cmake also removes modules that no longer exist in src/python.)
SET(POLYMERFTS_STAGE_DIR "${CMAKE_BINARY_DIR}/python/polymerfts")
ADD_CUSTOM_TARGET(polymerfts_stage ALL
    COMMAND ${CMAKE_COMMAND}
        -DSRC=${PROJECT_SOURCE_DIR}/src/python -DDST=${POLYMERFTS_STAGE_DIR}
        -DCORE=$<TARGET_FILE:_core> -P ${PROJECT_SOURCE_DIR}/cmake/stage_python.cmake
    DEPENDS _core
    COMMENT "Staging polymerfts package in ${POLYMERFTS_STAGE_DIR}"
    VERBATIM)

# ============================================================================
#  Install rules
# ============================================================================
IF(SKBUILD)
    # Building a wheel via scikit-build-core (pip install .): relative
    # destinations are mapped into the wheel's platlib automatically.
    SET(POLYMERFTS_PY_INSTALL_DIR "polymerfts")
ELSE()
    # Direct 'make install': query the SAME interpreter found by
    # FIND_PACKAGE(Python3). A bare "python" may be a different (system)
    # interpreter and would install into the wrong site-packages.
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

IF(POLYMERFTS_INSTALL_PYTHON)
    # Compiled extension
    INSTALL(TARGETS _core
            LIBRARY DESTINATION ${POLYMERFTS_PY_INSTALL_DIR}
            ARCHIVE DESTINATION ${POLYMERFTS_PY_INSTALL_DIR}
            COMPONENT python)
    # Pure-Python files (__init__.py, scft.py, lfts.py, ...)
    INSTALL(DIRECTORY src/python/
            DESTINATION ${POLYMERFTS_PY_INSTALL_DIR}
            COMPONENT python
            FILES_MATCHING PATTERN "*.py")
ENDIF()

IF(POLYMERFTS_INSTALL_HEADERS)
    # Headers for downstream C++ projects
    INSTALL(DIRECTORY src/common/
            DESTINATION include/polymerfts
            COMPONENT development
            FILES_MATCHING PATTERN "*.h")

    # CMake package config
    INCLUDE(CMakePackageConfigHelpers)
    WRITE_BASIC_PACKAGE_VERSION_FILE(
        "${CMAKE_CURRENT_BINARY_DIR}/polymerfts-config-version.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    CONFIGURE_PACKAGE_CONFIG_FILE(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/polymerfts-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/polymerfts-config.cmake"
        INSTALL_DESTINATION lib/cmake/polymerfts
    )
    INSTALL(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/polymerfts-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/polymerfts-config-version.cmake"
        DESTINATION lib/cmake/polymerfts
        COMPONENT development
    )
ENDIF()

# ============================================================================
#  Tests
# ============================================================================
IF(POLYMERFTS_BUILD_TESTS)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/tests)
ENDIF()
