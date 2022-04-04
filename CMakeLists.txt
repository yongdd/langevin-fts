# CMake
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(LANGEVIN_FTS CXX)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CUDA_STANDARD 14)
SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug  )

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

#MESSAGE("${CMAKE_SOURCE_DIR}")
#MESSAGE("${PROJECT_SOURCE_DIR}")
#MESSAGE("${PROJECT_BINARY_DIR}")
#MESSAGE("${CMAKE_CURRENT_SOURCE_DIR}")
#MESSAGE("${CMAKE_CURRENT_LIST_DIR}")
#SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
#SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})

#---------- FTS_LIBRARIES -----------

IF(CMAKE_BUILD_TYPE STREQUAL Debug)
    #IF(MSVC)
    #    ADD_COMPILE_OPTIONS(/W4 /WX)
    #ELSE()
    	ADD_COMPILE_OPTIONS(-Wall -Wextra)
    #ENDIF()
ENDIF()

#  OpenMP
FIND_PACKAGE(OpenMP)
IF (OPENMP_FOUND)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
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
    )

#  Intel MKL
FIND_PACKAGE(MKL)
IF(MKL_FOUND AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    SET(BUILD_CPU_MKL_LIB TRUE)
ELSE()
    SET(BUILD_CPU_MKL_LIB FALSE)
ENDIF()
IF(BUILD_CPU_MKL_LIB)
    ADD_DEFINITIONS(-DUSE_CPU_MKL)
    ADD_LIBRARY(cpu-mkl
        platforms/cpu/src/MklFFT1D.cpp
        platforms/cpu/src/MklFFT2D.cpp
        platforms/cpu/src/MklFFT3D.cpp
    	platforms/cpu/src/CpuPseudoGaussian.cpp
        platforms/cpu/src/CpuPseudoDiscrete.cpp
    	platforms/cpu/src/CpuAndersonMixing.cpp
        platforms/cpu/src/MklFactory.cpp
        )
ENDIF()

#  FFTW 
FIND_PATH(FFTW_INCLUDE NAMES fftw3.h)
FIND_LIBRARY(FFTW_LIBRARY NAMES fftw3)
MESSAGE("${FFTW_INCLUDE}")
MESSAGE("${FFTW_LIBRARY}")
IF(FFTW_INCLUDE AND FFTW_LIBRARY)
    SET(BUILD_CPU_FFTW_LIB TRUE)
ELSE()
    SET(BUILD_CPU_FFTW_LIB FALSE)
ENDIF()
IF(BUILD_CPU_FFTW_LIB)
    INCLUDE_DIRECTORIES(${FFTW_INCLUDE})
    ADD_DEFINITIONS(-DUSE_CPU_FFTW)
    ADD_LIBRARY(cpu-fftw
        platforms/cpu/src/FftwFFT1D.cpp
        platforms/cpu/src/FftwFFT2D.cpp
        platforms/cpu/src/FftwFFT3D.cpp
    	platforms/cpu/src/CpuPseudoGaussian.cpp
        platforms/cpu/src/CpuPseudoDiscrete.cpp
    	platforms/cpu/src/CpuAndersonMixing.cpp
        platforms/cpu/src/FftwFactory.cpp
    	)
ENDIF()

#  NVIDIA CUDA
FIND_PACKAGE(CUDAToolkit)
IF(CUDAToolkit_FOUND)
    ADD_DEFINITIONS(-DUSE_CUDA)
    ADD_LIBRARY(cuda
        platforms/cuda/src/CudaCommon.cu
        platforms/cuda/src/CudaSimulationBox.cu
        platforms/cuda/src/CudaPseudoGaussian.cu
        platforms/cuda/src/CudaPseudoDiscrete.cu
        platforms/cuda/src/CudaCircularBuffer.cu
        platforms/cuda/src/CudaAndersonMixing.cu
        platforms/cuda/src/CudaFactory.cu
        )
    SET_PROPERTY(TARGET cuda PROPERTY CUDA_ARCHITECTURES OFF)
ENDIF()

IF( (NOT BUILD_CPU_MKL_LIB) AND
    (NOT BUILD_CPU_FFTW_LIB) AND
    (NOT CUDAToolkit_FOUND) )
    MESSAGE( FATAL_ERROR "Could not find any FFT library, CMake will exit." )
ENDIF()

#  Factory
ADD_LIBRARY(factory
    main/src/PlatformSelector.cpp
    )

#---------- Python Wrapper -----------

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

#MESSAGE("${Python3_EXECUTABLE}")
#MESSAGE("${Python3_INCLUDE_DIRS}")
#MESSAGE("${Python3_NumPy_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES(${Python3_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Python3_NumPy_INCLUDE_DIRS})

SET_SOURCE_FILES_PROPERTIES(main/swig/langevinfts.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_LIBRARY(langevinfts
    LANGUAGE python
    SOURCES main/swig/langevinfts.i)

SWIG_LINK_LIBRARIES(langevinfts
	factory
	$<IF:$<BOOL:${CUDAToolkit_FOUND}>,cuda,>
	$<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cuda_driver,>
	$<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cufft,>
	$<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
	$<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-mkl,>
	$<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,cpu-fftw,>
	$<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,${FFTW_LIBRARY},>
	common
	)

EXECUTE_PROCESS(COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/_langevinfts.so DESTINATION ${PYTHON_SITE_PACKAGES})
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/langevinfts.py DESTINATION ${PYTHON_SITE_PACKAGES})
    
#---------- Test -----------

ENABLE_TESTING()
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/tests)
