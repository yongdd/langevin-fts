ENABLE_TESTING()

# Copy test data files to build directory
file(GLOB TEST_DATA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Stress*.txt")
file(COPY ${TEST_DATA_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Copy test data directory (for space group tests)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Basic tests for verification (~56 tests, ~40 sec total)
# Run with: ctest -L basic
SET(BASIC_TESTS
    # Platform & utilities
    TestDisplayArchitecture
    TestArray
    TestCircularBuffer
    TestScheduler
    TestPropagatorCode
    TestPropagatorCodeReduction
    TestCompareBranchKey
    TestContourLengthMapping

    # CPU FFT (all types)
    TestCpuFFT1D
    TestCpuFFT1DComplex
    TestCpuFFT2D
    TestCpuFFT2DComplex
    TestCpuFFT3D
    TestCpuFFT3DComplex
    TestCpuFFTMixedBC
    TestCpuSimulationBox
    TestCpuTridiagonal

    # CUDA FFT & utilities
    TestCudaFFT
    TestCudaFFTMixedBC
    TestCudaDCT
    TestCudaDST
    TestCudaSimulationBox
    TestCudaSimulationBoxComplex
    TestCudaTridiagonal
    TestCudaPinnedMemory
    TestCudaCubReduceSum
    TestCudaCubReduceSumClass
    TestCrysFFT
    TestCrysFFTObliqueZGlide

    # Propagator solvers
    TestCrankNicolson1D
    TestCrankNicolson2D
    TestCrankNicolson3D
    TestDCT2Propagator
    TestDST2Propagator
    TestPseudoLinearContinuous3D
    TestPseudoLinearDiscrete3D
    TestRealSpaceLinearContinuous3D
    TestMultipleLocalDs
    TestInitialConditionPropagator

    # Aggregation tests
    TestAggregation3D
    TestAggregationDendritic
    TestAggregationDiscreteCpu
    TestAggregationSideChains

    # Branched polymer tests
    TestPseudoBranchedContinuous3D
    TestPseudoBranchedDiscrete3D
    TestPseudoBranchedMixtureContinuous3D
    TestPseudoBranchedMixtureDiscrete3D

    # SCFT & Stress
    TestScft1D
    TestScft2D
    TestStress1D
    TestStress1DComplex
    TestStressDiscreteCpu1D
    TestStressDiscreteCuda1D

    # Space Group
    TestSpaceGroupReducedBasis
    TestMklRealTransform
    TestMklFftwAccuracyAllTypes
)

# Helper function to set test labels
FUNCTION(SET_TEST_LABELS TEST_NAME)
    IF(${TEST_NAME} IN_LIST BASIC_TESTS)
        SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES LABELS "basic")
    ENDIF()
ENDFUNCTION()

# Add Python Files
FILE(GLOB FILE_LIST "Test*.py")
FOREACH(FILE_NAME ${FILE_LIST})
    GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)
    ADD_TEST(NAME ${TEST_NAME} COMMAND ${Python3_EXECUTABLE} ${FILE_NAME})
    SET_TEST_LABELS(${TEST_NAME})
ENDFOREACH()

# Generate files named "Test*.cpp"
FILE(GLOB FILE_LIST "Test*.cpp")
FOREACH(FILE_NAME ${FILE_LIST})
    GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)
    IF ((${TEST_NAME} MATCHES "Cuda") AND NOT CUDAToolkit_FOUND)
        CONTINUE()
    ELSEIF((${TEST_NAME} MATCHES "Cpu") AND NOT BUILD_CPU_FFTW_LIB)
        CONTINUE()
    ELSEIF((${TEST_NAME} MATCHES "Fftw") AND NOT BUILD_CPU_FFTW_LIB)
        CONTINUE()
    ELSEIF((${TEST_NAME} MATCHES "Mkl") AND NOT BUILD_CPU_MKL_LIB)
        CONTINUE()
    ENDIF()

    ADD_EXECUTABLE(${TEST_NAME} ${FILE_NAME})
    TARGET_LINK_LIBRARIES(${TEST_NAME} PRIVATE
        factory
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,cuda,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cufft,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cusparse,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cusolver,>
        $<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,cpu-fftw,>
        $<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,-lfftw3,>
        $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
        common
        )

    IF(${TEST_NAME} MATCHES "ParamParser" OR ${TEST_NAME} MATCHES "Scft" OR ${TEST_NAME} MATCHES "Stress" OR ${TEST_NAME} MATCHES "Hexagonal")
        ADD_TEST(NAME ${TEST_NAME}
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
    ELSE()
        ADD_TEST(${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
    ENDIF()
    SET_TEST_LABELS(${TEST_NAME})
ENDFOREACH()

# Benchmark executables (not added to ctest)
FILE(GLOB BENCHMARK_LIST "Benchmark*.cpp")
FOREACH(FILE_NAME ${BENCHMARK_LIST})
    GET_FILENAME_COMPONENT(BENCH_NAME ${FILE_NAME} NAME_WE)
    # Skip platform-specific benchmarks if platform is not available
    IF((${BENCH_NAME} MATCHES "Cuda") AND NOT CUDAToolkit_FOUND)
        CONTINUE()
    ELSEIF((${BENCH_NAME} MATCHES "Mkl") AND NOT BUILD_CPU_MKL_LIB)
        CONTINUE()
    ELSEIF((${BENCH_NAME} MATCHES "Fftw") AND NOT BUILD_CPU_FFTW_LIB)
        CONTINUE()
    ENDIF()

    ADD_EXECUTABLE(${BENCH_NAME} ${FILE_NAME})
    TARGET_LINK_LIBRARIES(${BENCH_NAME} PRIVATE
        factory
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,cuda,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cufft,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cusparse,>
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cusolver,>
        $<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,cpu-fftw,>
        $<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,-lfftw3,>
        $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
        common
        )
ENDFOREACH()

IF(CUDAToolkit_FOUND)
    # Generate files named "Test*.cu"
    FILE(GLOB FILE_LIST "Test*.cu")
    FOREACH(FILE_NAME ${FILE_LIST})
        GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)

        ADD_EXECUTABLE(${TEST_NAME} ${FILE_NAME})
        TARGET_LINK_LIBRARIES(${TEST_NAME} PRIVATE
            factory cuda CUDA::cufft CUDA::cusparse CUDA::cusolver
            $<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,cpu-fftw,>
            $<IF:$<BOOL:${BUILD_CPU_FFTW_LIB}>,-lfftw3,>
            $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
            common)

        SET_PROPERTY(TARGET ${TEST_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
        ADD_TEST(${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
        SET_TEST_LABELS(${TEST_NAME})
    ENDFOREACH()
ENDIF()
