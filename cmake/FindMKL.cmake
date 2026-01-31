# FindMKL.cmake - Fallback MKL finder for systems without MKLConfig.cmake
#
# This module finds Intel MKL and sets:
#   MKL_FOUND - True if MKL was found
#   MKL_INCLUDE_DIRS - MKL include directories
#   MKL_LIBRARIES - MKL libraries to link
#
# It also creates the MKL::MKL imported target if MKL is found.

if(MKL_FOUND)
    return()
endif()

# Search paths - check environment variables and common locations
set(_MKL_SEARCH_PATHS
    ${MKL_ROOT}
    $ENV{MKLROOT}
    $ENV{CONDA_PREFIX}
    /opt/intel/oneapi/mkl/latest
    /opt/intel/mkl
    /usr/local
)

# Find include directory
find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    PATHS ${_MKL_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Find core library
find_library(MKL_CORE_LIB
    NAMES mkl_core
    PATHS ${_MKL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/intel64
)

# Find interface library (LP64 for 32-bit integers, ILP64 for 64-bit)
find_library(MKL_INTEL_LP64_LIB
    NAMES mkl_intel_lp64
    PATHS ${_MKL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/intel64
)

# Find threading library - prefer sequential for simplicity
find_library(MKL_SEQUENTIAL_LIB
    NAMES mkl_sequential
    PATHS ${_MKL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/intel64
)

# Alternative: Intel threading (requires libiomp5)
find_library(MKL_INTEL_THREAD_LIB
    NAMES mkl_intel_thread
    PATHS ${_MKL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/intel64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL
    REQUIRED_VARS MKL_INCLUDE_DIR MKL_INTEL_LP64_LIB MKL_CORE_LIB
    FAIL_MESSAGE "Could not find Intel MKL. Set MKL_ROOT or MKLROOT environment variable."
)

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})

    # Determine which threading library to use
    if(MKL_SEQUENTIAL_LIB)
        set(_MKL_THREADING_LIB ${MKL_SEQUENTIAL_LIB})
    elseif(MKL_INTEL_THREAD_LIB)
        set(_MKL_THREADING_LIB ${MKL_INTEL_THREAD_LIB})
    else()
        message(FATAL_ERROR "MKL threading library not found")
    endif()

    set(MKL_LIBRARIES
        ${MKL_INTEL_LP64_LIB}
        ${_MKL_THREADING_LIB}
        ${MKL_CORE_LIB}
        dl
        pthread
        m
    )

    # Create imported target
    if(NOT TARGET MKL::MKL)
        add_library(MKL::MKL INTERFACE IMPORTED)
        set_target_properties(MKL::MKL PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${MKL_LIBRARIES}"
        )
    endif()

    # Extract library directory for RPATH
    get_filename_component(MKL_LIBRARY_DIR ${MKL_CORE_LIB} DIRECTORY)
    set(MKL_LIBRARY_DIRS ${MKL_LIBRARY_DIR} CACHE PATH "MKL library directory")

    message(STATUS "Found MKL: ${MKL_LIBRARY_DIR}")
    message(STATUS "  Include: ${MKL_INCLUDE_DIRS}")
endif()

mark_as_advanced(
    MKL_INCLUDE_DIR
    MKL_CORE_LIB
    MKL_INTEL_LP64_LIB
    MKL_SEQUENTIAL_LIB
    MKL_INTEL_THREAD_LIB
)
