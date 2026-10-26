# 3.18: FIND_PACKAGE(Python3 ... Development.Module); the GPU backend needs 3.25.2 (checked below)
CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

# ============================================================================
#  Project
# ============================================================================
SET(POLYMERFTS_VERSION_MAJOR 0)
SET(POLYMERFTS_VERSION_MINOR 10)
SET(POLYMERFTS_VERSION_PATCH 0)
SET(POLYMERFTS_VERSION "${POLYMERFTS_VERSION_MAJOR}.${POLYMERFTS_VERSION_MINOR}.${POLYMERFTS_VERSION_PATCH}")

PROJECT(LANGEVIN_FTS
    VERSION ${POLYMERFTS_VERSION}
    DESCRIPTION "Polymer Field Theory Simulation Library for SCFT, L-FTS, and CL-FTS"
    LANGUAGES CXX
)

# Custom find modules (FindMKL.cmake)
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# conda on Windows keeps native libraries under <prefix>/Library, so
# -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" works there as on Linux.
IF(WIN32)
    FOREACH(_prefix ${CMAKE_PREFIX_PATH})
        IF(IS_DIRECTORY "${_prefix}/Library")
            LIST(APPEND CMAKE_PREFIX_PATH "${_prefix}/Library")
        ENDIF()
    ENDFOREACH()
ENDIF()

# ============================================================================
#  Options
# ============================================================================
OPTION(POLYMERFTS_USE_FFTW        "Enable FFTW3 backend (GPL license)"        OFF)
OPTION(POLYMERFTS_USE_MKL         "Enable Intel MKL backend"                  OFF)
# AUTO: build the GPU backend when a usable CUDA toolchain is found (see the
# CUDA backend section), otherwise warn and build without it; ON: require it;
# OFF: never. Earlier versions declared it as a BOOL option whose ON meant what
# AUTO means now, so such a cache entry is converted (ON -> AUTO, OFF -> OFF).
GET_PROPERTY(_cuda_option_type CACHE POLYMERFTS_USE_CUDA PROPERTY TYPE)
IF(_cuda_option_type STREQUAL "BOOL")
    IF(POLYMERFTS_USE_CUDA)
        SET(_cuda_option_value AUTO)
    ELSE()
        SET(_cuda_option_value OFF)
    ENDIF()
    SET(POLYMERFTS_USE_CUDA ${_cuda_option_value} CACHE STRING "CUDA backend: AUTO, ON or OFF" FORCE)
ENDIF()
SET(POLYMERFTS_USE_CUDA AUTO CACHE STRING "CUDA backend: AUTO, ON or OFF")
SET_PROPERTY(CACHE POLYMERFTS_USE_CUDA PROPERTY STRINGS AUTO ON OFF)
OPTION(POLYMERFTS_BUILD_TESTS     "Build test executables"                    ON)
OPTION(POLYMERFTS_INSTALL_PYTHON  "Install Python module"                     ON)
OPTION(POLYMERFTS_INSTALL_HEADERS "Install C++ headers for downstream projects" OFF)
OPTION(POLYMERFTS_WERROR          "Treat host C++ warnings as errors (CI)"    OFF)
SET(POLYMERFTS_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA compute capabilities to compile for (semicolon-separated; empty: a default for the toolkit, see the CUDA backend section)")

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
# against (e.g. the CUDA lib), so importing it does not need LD_LIBRARY_PATH.
# Directories the compiler searches by itself (e.g. through LIBRARY_PATH) are not
# recorded; the conda environment's lib is then found through the Python
# interpreter's own RPATH ($ORIGIN/../lib).
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Warnings for host code in every build type; Debug adds -Wextra and gprof
# instrumentation. (For AddressSanitizer add -fsanitize=address
# -fno-omit-frame-pointer to CMAKE_CXX_FLAGS on the command line.)
IF(MSVC)
    # /W3 without the implicit-narrowing warnings C4244 and C4267, which GCC's
    # -Wall leaves out too. The sources are UTF-8 (Greek letters in comments),
    # which MSVC does not assume under a non-UTF-8 code page; the same holds for
    # the host compiler of nvcc. The CUDA headers (CCCL) need the conforming
    # preprocessor. <cmath> defines M_PI only with _USE_MATH_DEFINES; the CRT
    # marks getenv and the POSIX names (dup, fileno) as deprecated.
    ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
                        $<$<COMPILE_LANGUAGE:CXX>:/wd4267> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
                        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    ADD_COMPILE_DEFINITIONS(_USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
    IF(POLYMERFTS_WERROR)
        ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:/WX>)
    ENDIF()
ELSE()
    ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
    IF(POLYMERFTS_WERROR)
        ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    ENDIF()
    IF(CMAKE_BUILD_TYPE STREQUAL Debug)
        # host code only: nvcc rejects -Wextra
        ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-pg>)
        ADD_LINK_OPTIONS(-pg)
    ENDIF()
ENDIF()

# OpenMP (propagator-level parallelism on the CPU; the CUDA target adds the
# OpenMP flag for its host code below). MSVC's default OpenMP 2.0 rejects the
# unsigned loop indices of the propagator loops; its LLVM runtime (CMake 3.30+)
# accepts them.
IF(MSVC AND CMAKE_VERSION VERSION_LESS 3.30)
    MESSAGE(FATAL_ERROR "MSVC builds need CMake 3.30 or newer for the LLVM OpenMP runtime "
                        "(/openmp:llvm), found ${CMAKE_VERSION}. Install a newer CMake, "
                        "e.g. 'conda install -c conda-forge \"cmake>=3.30\"'.")
ENDIF()
SET(OpenMP_RUNTIME_MSVC llvm)
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
    # The upstream Python wheels also ship the native library and headers.
    # pip's isolated build hides the target environment's Python packages.
    # Also look in that interpreter's install directories for the native
    # dependency, just as we search the conda prefix for other libraries.
    FIND_PACKAGE(Python3 3.11 COMPONENTS Interpreter REQUIRED)
    EXECUTE_PROCESS(COMMAND ${Python3_EXECUTABLE} -c
        [=[
import importlib.util
import pathlib
import sysconfig
spec = importlib.util.find_spec('spglib')
roots = [pathlib.Path(sysconfig.get_path(name)) / 'spglib' for name in ('purelib', 'platlib')]
if spec and spec.origin:
    roots.insert(0, pathlib.Path(spec.origin).parent)
print(';'.join(dict.fromkeys(str(root) for root in roots)))
]=]
        OUTPUT_VARIABLE _spglib_python_roots OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    FIND_LIBRARY(SPGLIB_LIBRARIES NAMES symspg spglib
        HINTS ENV SPGLIB_ROOT ${_spglib_python_roots} PATH_SUFFIXES lib lib64)
    FIND_PATH(SPGLIB_INCLUDE_DIRS spglib.h
        HINTS ENV SPGLIB_ROOT ${_spglib_python_roots} PATH_SUFFIXES include)
    IF(SPGLIB_LIBRARIES AND SPGLIB_INCLUDE_DIRS)
        SET(SPGLIB_FOUND TRUE)
    ENDIF()
ENDIF()
IF(SPGLIB_FOUND)
    INCLUDE_DIRECTORIES(${SPGLIB_INCLUDE_DIRS})
    MESSAGE(STATUS "spglib found: ${SPGLIB_LIBRARIES}")
ELSE()
    MESSAGE(FATAL_ERROR "spglib is required but was not found. "
        "Install with: python -m pip install spglib "
        "(or apt install libsymspg-dev), or set SPGLIB_ROOT.")
ENDIF()

# LAPACKE: SVD solver of Anderson mixing. OpenBLAS can provide its C interface
# directly; conda-forge's openblas supplies lapacke.h. Keep standalone LAPACKE
# supported, and use MKL only when its backend was explicitly requested.
FIND_LIBRARY(LAPACKE_LIBRARY NAMES lapacke openblas
    HINTS ENV LAPACKE_ROOT PATH_SUFFIXES lib lib64)
IF(NOT LAPACKE_LIBRARY AND POLYMERFTS_USE_MKL)
    FIND_PACKAGE(MKL QUIET)
    IF(MKL_FOUND)
        SET(LAPACKE_LIBRARY ${MKL_LIBRARIES})
    ENDIF()
ENDIF()
FIND_PATH(LAPACKE_INCLUDE_DIR NAMES lapacke.h mkl_lapacke.h
    HINTS ENV LAPACKE_ROOT ENV MKLROOT PATH_SUFFIXES include include/openblas)
IF(LAPACKE_LIBRARY AND LAPACKE_INCLUDE_DIR)
    # Some OpenBLAS builds omit LAPACKE. Check both SVD entry points with the
    # selected headers/library before starting the main build.
    INCLUDE(CheckCXXSourceCompiles)
    INCLUDE(CMakePushCheckState)
    CMAKE_PUSH_CHECK_STATE(RESET)
    SET(CMAKE_REQUIRED_INCLUDES ${LAPACKE_INCLUDE_DIR})
    SET(CMAKE_REQUIRED_LIBRARIES ${LAPACKE_LIBRARY})
    UNSET(POLYMERFTS_LAPACKE_HAS_SVD CACHE)
    CHECK_CXX_SOURCE_COMPILES("
        #include <complex>
        #ifdef _MSC_VER
        #define lapack_complex_float std::complex<float>
        #define lapack_complex_double std::complex<double>
        #endif
        #if __has_include(<lapacke.h>)
        #include <lapacke.h>
        #else
        #include <mkl_lapacke.h>
        #endif
        int main() {
            lapack_int rank = 0;
            double a = 1.0, b = 1.0, s = 0.0;
            lapack_complex_double z_a{}, z_b{};
            return LAPACKE_dgelsd(LAPACK_ROW_MAJOR, 1, 1, 1,
                                 &a, 1, &b, 1, &s, 1e-10, &rank)
                 + LAPACKE_zgelsd(LAPACK_ROW_MAJOR, 1, 1, 1,
                                 &z_a, 1, &z_b, 1, &s, 1e-10, &rank);
        }" POLYMERFTS_LAPACKE_HAS_SVD)
    CMAKE_POP_CHECK_STATE()
    IF(NOT POLYMERFTS_LAPACKE_HAS_SVD)
        MESSAGE(FATAL_ERROR "The selected LAPACKE headers/library do not provide "
            "LAPACKE_dgelsd and LAPACKE_zgelsd. Install OpenBLAS with LAPACKE support "
            "(conda install --override-channels -c conda-forge openblas), "
            "or set LAPACKE_LIBRARY and LAPACKE_INCLUDE_DIR to a matching provider.")
    ENDIF()
    INCLUDE_DIRECTORIES(${LAPACKE_INCLUDE_DIR})
    MESSAGE(STATUS "LAPACKE found: ${LAPACKE_LIBRARY} (${LAPACKE_INCLUDE_DIR})")
ELSE()
    MESSAGE(FATAL_ERROR "LAPACKE (library and lapacke.h or mkl_lapacke.h) is required but was not found. "
        "Install with: conda install --override-channels -c conda-forge openblas, "
        "or apt install liblapacke-dev, or set LAPACKE_ROOT. "
        "MKL can also supply LAPACKE when POLYMERFTS_USE_MKL=ON.")
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
    src/common/AbstractFactory.cpp
    src/common/Scheduler.cpp
    src/common/SpaceGroup.cpp
    src/common/FourierTransform.cpp
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
    src/platforms/cpu/CpuFourierTransform.cpp
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
            "Install fftw (e.g. conda install --override-channels -c conda-forge fftw) or set FFTW_ROOT.")
    ENDIF()

    SET(BUILD_CPU_FFTW_LIB TRUE)
    ADD_DEFINITIONS(-DUSE_CPU_FFTW)
    # fftw3.h imports the data of the FFTW DLL (fftw_version) only with FFTW_DLL;
    # without it, the linker takes fftw_version from MKL's FFTW wrappers
    IF(WIN32)
        ADD_DEFINITIONS(-DFFTW_DLL)
    ENDIF()
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
            "Install MKL (e.g. conda install --override-channels -c conda-forge mkl-devel) "
            "or set MKLROOT.")
    ENDIF()

    SET(BUILD_CPU_MKL_LIB TRUE)
    ADD_DEFINITIONS(-DUSE_CPU_MKL)
    # Other dependencies may share an include directory with a different MKL.
    # Use the headers that belong to the selected MKL libraries first.
    INCLUDE_DIRECTORIES(BEFORE ${MKL_INCLUDE_DIRS})
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
# The CUDA code is C++20. Building it needs nvcc 12.2+ (11.x has no -std=c++20;
# 12.0/12.1 cannot evaluate std::source_location::current(), which CudaBuffer
# uses to name the caller of a failed allocation), CMake 3.25.2+ (which passes
# -std=c++20 to nvcc) and std::source_location in both the C++ compiler and
# nvcc's host compiler (GCC 11+, Clang 15+). Every check looks at the nvcc that
# compiles the code. With POLYMERFTS_USE_CUDA=AUTO a missing piece is a warning
# and the GPU backend is left out; with ON it stops the configure step.
STRING(TOUPPER "${POLYMERFTS_USE_CUDA}" _cuda_request)
IF(NOT _cuda_request STREQUAL "AUTO")
    IF(NOT _cuda_request MATCHES "^(ON|OFF|TRUE|FALSE|YES|NO|Y|N|1|0)$")
        MESSAGE(FATAL_ERROR "POLYMERFTS_USE_CUDA must be AUTO, ON or OFF, got '${POLYMERFTS_USE_CUDA}'")
    ELSEIF(POLYMERFTS_USE_CUDA)
        SET(_cuda_request ON)
    ELSE()
        SET(_cuda_request OFF)
    ENDIF()
ENDIF()
SET(POLYMERFTS_BUILD_CUDA FALSE)

# Report why the GPU backend cannot be built: an error with ON, a warning with
# AUTO. The reason may be given in several string arguments; they are joined
# one by one (${ARGV} would split them at the semicolons inside).
FUNCTION(polymerfts_cuda_unusable)
    SET(_reason "")
    MATH(EXPR _last "${ARGC} - 1")
    FOREACH(_i RANGE 0 ${_last})
        STRING(APPEND _reason "${ARGV${_i}}")
    ENDFOREACH()
    IF(_cuda_request STREQUAL "ON")
        MESSAGE(FATAL_ERROR "POLYMERFTS_USE_CUDA=ON, but ${_reason}")
    ELSE()
        MESSAGE(WARNING "Building without the GPU backend: ${_reason} "
            "(POLYMERFTS_USE_CUDA=AUTO; set it to ON to make this an error, or OFF to silence it)")
    ENDIF()
ENDFUNCTION()

IF(NOT _cuda_request STREQUAL "OFF")
    # The nvcc CMake would use: CUDACXX, else the first on PATH (for the messages)
    IF(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
        SET(_nvcc_candidate "$ENV{CUDACXX}")
    ELSE()
        FIND_PROGRAM(_nvcc_candidate nvcc NO_CACHE)
    ENDIF()
    # CHECK_LANGUAGE caches a failed search as NOTFOUND and then never searches
    # again; search again when an nvcc is around now
    IF(DEFINED CACHE{CMAKE_CUDA_COMPILER} AND NOT CMAKE_CUDA_COMPILER AND _nvcc_candidate)
        UNSET(CMAKE_CUDA_COMPILER CACHE)
    ENDIF()
    # CHECK_LANGUAGE runs nvcc in a sub-project that does not see a host compiler
    # given as -DCMAKE_CUDA_HOST_COMPILER, and its (empty) result would then hide
    # the cache entry from ENABLE_LANGUAGE: hand it over through CUDAHOSTCXX and
    # drop the normal variable afterwards.
    IF(CMAKE_CUDA_HOST_COMPILER AND "$ENV{CUDAHOSTCXX}" STREQUAL "")
        SET(ENV{CUDAHOSTCXX} "${CMAKE_CUDA_HOST_COMPILER}")
    ENDIF()
    INCLUDE(CheckLanguage)
    CHECK_LANGUAGE(CUDA)          # nvcc from CMAKE_CUDA_COMPILER, CUDACXX or PATH
    UNSET(CMAKE_CUDA_HOST_COMPILER)
    IF(NOT CMAKE_CUDA_COMPILER AND _nvcc_candidate)
        polymerfts_cuda_unusable("${_nvcc_candidate} did not compile a test program, usually "
            "because its host compiler is not supported (nvcc 12.2 rejects GCC 13): set "
            "CUDAHOSTCXX or -DCMAKE_CUDA_HOST_COMPILER to a supported compiler "
            "(CMakeFiles/CMakeConfigureLog.yaml has the output)")
    ELSEIF(NOT CMAKE_CUDA_COMPILER)
        polymerfts_cuda_unusable("no CUDA compiler (nvcc) was found on PATH, in CUDACXX or in "
            "CMAKE_CUDA_COMPILER")
    ELSEIF(CMAKE_VERSION VERSION_LESS 3.25.2)
        polymerfts_cuda_unusable("CMake ${CMAKE_VERSION} cannot compile C++20 CUDA code; the "
            "GPU backend needs CMake 3.25.2 or newer (environment.yml installs 3.31)")
    ELSE()
        EXECUTE_PROCESS(COMMAND "${CMAKE_CUDA_COMPILER}" --version
            OUTPUT_VARIABLE _nvcc_output ERROR_QUIET)
        STRING(REGEX MATCH "release ([0-9]+[.][0-9]+)" _nvcc_release "${_nvcc_output}")
        SET(_nvcc_version "${CMAKE_MATCH_1}")
        IF(_nvcc_version STREQUAL "")
            SET(_nvcc_found "${CMAKE_CUDA_COMPILER}")
            UNSET(CMAKE_CUDA_COMPILER CACHE)    # search again next time
            polymerfts_cuda_unusable("the CUDA version of ${_nvcc_found} could not be read "
                "(`nvcc --version` gave no release number)")
        ELSEIF(_nvcc_version VERSION_LESS 12.2)
            SET(_nvcc_found "${CMAKE_CUDA_COMPILER}")
            # Not enabled yet, so the next configure may search again (e.g. after
            # a newer toolkit was put first on PATH)
            UNSET(CMAKE_CUDA_COMPILER CACHE)
            polymerfts_cuda_unusable("${_nvcc_found} is CUDA ${_nvcc_version}; the GPU backend "
                "needs CUDA 12.2 or newer (C++20 device code). Put a 12.2+ nvcc first on PATH "
                "(or in CUDACXX, or pass -DCMAKE_CUDA_COMPILER=...) and configure again")
        ELSE()
            SET(POLYMERFTS_BUILD_CUDA TRUE)
        ENDIF()
    ENDIF()
ENDIF()

IF(POLYMERFTS_BUILD_CUDA)
    ENABLE_LANGUAGE(CUDA)

    # The toolkit of that nvcc. A toolkit found by an earlier configure (cached)
    # would otherwise win over the compiler.
    GET_FILENAME_COMPONENT(_nvcc_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    GET_FILENAME_COMPONENT(_nvcc_bin "${_nvcc_bin}" REALPATH)
    IF(DEFINED CACHE{CUDAToolkit_BIN_DIR})
        GET_FILENAME_COMPONENT(_cached_bin "${CUDAToolkit_BIN_DIR}" REALPATH)
        IF(NOT _cached_bin STREQUAL _nvcc_bin)
            UNSET(CUDAToolkit_BIN_DIR CACHE)
            UNSET(CUDAToolkit_NVCC_EXECUTABLE CACHE)
        ENDIF()
    ENDIF()
    FIND_PACKAGE(CUDAToolkit QUIET)
    STRING(REGEX MATCH "^[0-9]+[.][0-9]+" _compiler_release "${CMAKE_CUDA_COMPILER_VERSION}")
    IF(NOT CUDAToolkit_FOUND)
        polymerfts_cuda_unusable("the CUDA toolkit of ${CMAKE_CUDA_COMPILER} was not found")
        SET(POLYMERFTS_BUILD_CUDA FALSE)
    ELSEIF(NOT "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}" STREQUAL _compiler_release)
        polymerfts_cuda_unusable("the CUDA toolkit found (${CUDAToolkit_VERSION} at "
            "${CUDAToolkit_TARGET_DIR}) is not the one of ${CMAKE_CUDA_COMPILER} "
            "(${CMAKE_CUDA_COMPILER_VERSION}); delete CMakeCache.txt and configure again")
        SET(POLYMERFTS_BUILD_CUDA FALSE)
    ELSEIF(NOT TARGET CUDA::cufft)
        polymerfts_cuda_unusable("the CUDA toolkit of ${CMAKE_CUDA_COMPILER} has no cuFFT library")
        SET(POLYMERFTS_BUILD_CUDA FALSE)
    ENDIF()
ENDIF()

IF(POLYMERFTS_BUILD_CUDA)
    # std::source_location in the host C++ compiler (CudaCommon.h is also
    # included by host sources) and in nvcc with its host compiler
    INCLUDE(CheckSourceCompiles)
    SET(_source_location_test "#include <source_location>
        int line(std::source_location where = std::source_location::current())
        { return static_cast<int>(where.line()); }
        int main() { return line() > 0 ? 0 : 1; }")
    CHECK_SOURCE_COMPILES(CXX "${_source_location_test}" POLYMERFTS_CXX_HAS_SOURCE_LOCATION)
    CHECK_SOURCE_COMPILES(CUDA "${_source_location_test}" POLYMERFTS_CUDA_HAS_SOURCE_LOCATION)
    IF(NOT POLYMERFTS_CXX_HAS_SOURCE_LOCATION OR NOT POLYMERFTS_CUDA_HAS_SOURCE_LOCATION)
        # Not kept as a cached failure, so a fixed toolchain is checked again
        UNSET(POLYMERFTS_CXX_HAS_SOURCE_LOCATION CACHE)
        UNSET(POLYMERFTS_CUDA_HAS_SOURCE_LOCATION CACHE)
        polymerfts_cuda_unusable("std::source_location does not compile with the C++ compiler "
            "(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}) or with nvcc "
            "${CMAKE_CUDA_COMPILER_VERSION} and its host compiler (${CMAKE_CUDA_HOST_COMPILER_ID} "
            "${CMAKE_CUDA_HOST_COMPILER_VERSION}); use GCC 11+ or Clang 15+ (for nvcc through "
            "CUDAHOSTCXX or -DCMAKE_CUDA_HOST_COMPILER)")
        SET(POLYMERFTS_BUILD_CUDA FALSE)
    ENDIF()
ENDIF()

IF(POLYMERFTS_BUILD_CUDA)
    MESSAGE(STATUS "CUDA toolkit ${CUDAToolkit_VERSION}: ${CUDAToolkit_TARGET_DIR}")

    # Compute capabilities: the user's list, or a default the toolkit supports
    # (CUDA 12.8 added sm_100 and sm_120, CUDA 13 dropped sm_60, sm_61 and sm_70).
    # A normal variable shadows the empty cache entry, so tests/CMakeLists.txt
    # sees the same list.
    IF(POLYMERFTS_CUDA_ARCHITECTURES STREQUAL "")
        IF(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
            SET(POLYMERFTS_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
        ELSEIF(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
            SET(POLYMERFTS_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90;100;120")
        ELSE()
            SET(POLYMERFTS_CUDA_ARCHITECTURES "75;80;86;89;90;100;120")
        ENDIF()
    ENDIF()
    MESSAGE(STATUS "CUDA architectures: ${POLYMERFTS_CUDA_ARCHITECTURES}")

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

    # Link the cuFFT of this toolkit. FindCUDAToolkit looks for the libraries on
    # CMAKE_PREFIX_PATH first, so a conda environment that carries another CUDA's
    # runtime libraries (e.g. cudatoolkit for CuPy) would otherwise supply a
    # cuFFT of another version. The toolkit's library directories are the ones
    # nvcc links from (without the driver stubs).
    SET(_toolkit_lib_dirs "")
    FOREACH(_dir ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES}
                 "${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64" "${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib"
                 "${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib/x64")
        IF(NOT _dir MATCHES "/stubs/?$")
            LIST(APPEND _toolkit_lib_dirs "${_dir}")
        ENDIF()
    ENDFOREACH()
    FIND_LIBRARY(_toolkit_cufft NAMES cufft PATHS ${_toolkit_lib_dirs} NO_DEFAULT_PATH NO_CACHE)
    IF(_toolkit_cufft)
        SET_TARGET_PROPERTIES(CUDA::cufft PROPERTIES IMPORTED_LOCATION "${_toolkit_cufft}")
    ELSE()
        GET_TARGET_PROPERTY(_found_cufft CUDA::cufft IMPORTED_LOCATION)
        MESSAGE(WARNING "No cuFFT in the library directories of ${CMAKE_CUDA_COMPILER} "
            "(${_toolkit_lib_dirs}); linking ${_found_cufft}, which may belong to another CUDA version.")
    ENDIF()
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
        src/platforms/cuda/CudaFourierTransform.cu
        src/platforms/cuda/CudaFactory.cu
    )
    SET_PROPERTY(TARGET cuda PROPERTY CUDA_ARCHITECTURES ${POLYMERFTS_CUDA_ARCHITECTURES})
    IF(MSVC)
        TARGET_COMPILE_OPTIONS(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp:llvm>)
    ELSE()
        TARGET_COMPILE_OPTIONS(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    ENDIF()
ENDIF()

IF(NOT BUILD_CPU_FFTW_LIB AND NOT BUILD_CPU_MKL_LIB AND NOT POLYMERFTS_BUILD_CUDA)
    MESSAGE(FATAL_ERROR "No computational backend enabled: turn on at least one of "
        "POLYMERFTS_USE_MKL, POLYMERFTS_USE_FFTW, or POLYMERFTS_USE_CUDA (with a usable CUDA "
        "12.2+ toolchain). For a pip build pass them as -C cmake.define.POLYMERFTS_USE_FFTW=ON.")
ENDIF()

# Platform selector: depends on every enabled backend, hence a separate target
ADD_LIBRARY(factory
    src/common/PlatformSelector.cpp
)

# Everything a consumer of the library links (the Python module, tests,
# benchmarks): selector, enabled backends, common core. Order matters for
# static libraries.
SET(POLYMERFTS_BACKEND_LIBS factory)
IF(POLYMERFTS_BUILD_CUDA)
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
SET(_pybind11_minimum 2.13)
IF(Python3_VERSION VERSION_GREATER_EQUAL "3.14")
    SET(_pybind11_minimum 3.0)
ENDIF()
FIND_PACKAGE(pybind11 ${_pybind11_minimum} REQUIRED HINTS ${_pybind11_cmakedir})

# pybind11::module carries the Python and pybind11 include directories
ADD_LIBRARY(_core MODULE src/pybind11/polymerfts_core.cpp)
IF(WIN32)
    SET_TARGET_PROPERTIES(_core PROPERTIES PREFIX "" SUFFIX ".pyd")
ELSE()
    SET_TARGET_PROPERTIES(_core PROPERTIES PREFIX "" SUFFIX ".so")
ENDIF()
TARGET_LINK_LIBRARIES(_core PUBLIC pybind11::module PRIVATE polymerfts_backends)

# Embed the provenance of this extension, including builds from source archives.
SET(POLYMERFTS_GIT_COMMIT "unknown")
SET(POLYMERFTS_SOURCE_MODIFIED "unknown")
FIND_PACKAGE(Git QUIET)
IF(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    EXECUTE_PROCESS(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE POLYMERFTS_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    EXECUTE_PROCESS(COMMAND ${GIT_EXECUTABLE} diff --quiet HEAD -- CMakeLists.txt cmake src
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} RESULT_VARIABLE _source_diff)
    IF(_source_diff EQUAL 0)
        SET(POLYMERFTS_SOURCE_MODIFIED "false")
    ELSE()
        SET(POLYMERFTS_SOURCE_MODIFIED "true")
    ENDIF()
ENDIF()
STRING(TOUPPER "${CMAKE_BUILD_TYPE}" _build_config)
SET(POLYMERFTS_HOST_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_config}}")
SET(POLYMERFTS_DEVICE_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${_build_config}}")
CONFIGURE_FILE(cmake/BuildInfo.h.in generated/BuildInfo.h @ONLY)
TARGET_INCLUDE_DIRECTORIES(_core PRIVATE ${CMAKE_BINARY_DIR}/generated)

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
    # Windows paths come with backslashes
    FILE(TO_CMAKE_PATH "${Python3_SITEARCH}" Python3_SITEARCH)
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

    # The static libraries and polymerfts::polymerfts_backends, which ties them
    # together, for find_package(polymerfts). OpenMP and the CUDA runtime reach
    # this build through compile flags and the CUDA language; an installed
    # consumer gets them as link dependencies instead.
    SET(_polymerfts_export_targets factory common polymerfts_backends)
    IF(POLYMERFTS_BUILD_CUDA)
        LIST(APPEND _polymerfts_export_targets cuda)
    ENDIF()
    IF(BUILD_CPU_FFTW_LIB)
        LIST(APPEND _polymerfts_export_targets cpu-fftw)
    ENDIF()
    IF(BUILD_CPU_MKL_LIB)
        LIST(APPEND _polymerfts_export_targets cpu-mkl)
    ENDIF()
    TARGET_INCLUDE_DIRECTORIES(common INTERFACE
        $<INSTALL_INTERFACE:include/polymerfts> $<INSTALL_INTERFACE:${SPGLIB_INCLUDE_DIRS}>)
    TARGET_LINK_LIBRARIES(polymerfts_backends INTERFACE $<INSTALL_INTERFACE:OpenMP::OpenMP_CXX>)
    IF(POLYMERFTS_BUILD_CUDA)
        TARGET_LINK_LIBRARIES(polymerfts_backends INTERFACE $<INSTALL_INTERFACE:CUDA::cudart_static>)
    ENDIF()
    INSTALL(TARGETS ${_polymerfts_export_targets}
            EXPORT polymerfts-targets
            ARCHIVE DESTINATION lib COMPONENT development)
    INSTALL(EXPORT polymerfts-targets
            NAMESPACE polymerfts::
            DESTINATION lib/cmake/polymerfts
            COMPONENT development)

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
