# Tests that read input files (Stress*/Hexagonal*/Scft*) run with the source
# directory as working directory; the Python tests locate tests/data from
# their own path. Nothing is copied into the build tree.

# Basic tests for verification (label "basic", about one minute in total)
# Run with: ctest -L basic
SET(BASIC_TESTS
    # Parameter handling of the simulation classes and array handling of the
    # bindings (CPU, a few seconds)
    TestParams
    TestComplexMask
    # Platform & utilities
    TestDisplayArchitecture
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

# Add Python Files. They import the package staged in the build tree
# (see polymerfts_stage in the top-level CMakeLists.txt), so 'make install'
# is not required and an old site-packages install is never tested by mistake.
# Only the staged package is on PYTHONPATH; the configuring shell's value is
# not baked in.
FILE(GLOB FILE_LIST "Test*.py")
FOREACH(FILE_NAME ${FILE_LIST})
    GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)
    ADD_TEST(NAME ${TEST_NAME} COMMAND ${Python3_EXECUTABLE} ${FILE_NAME})
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
    SET_TEST_LABELS(${TEST_NAME})
ENDFOREACH()

# Generate files named "Test*.cpp"
FILE(GLOB FILE_LIST "Test*.cpp")
FOREACH(FILE_NAME ${FILE_LIST})
    GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)
    IF ((${TEST_NAME} MATCHES "Cuda") AND NOT CUDAToolkit_FOUND)
        CONTINUE()
    # "Cpu" tests are backend-agnostic (sources select FFTW/MKL via #ifdef),
    # so they only need SOME CPU backend, not FFTW specifically.
    ELSEIF((${TEST_NAME} MATCHES "Cpu") AND NOT BUILD_CPU_FFTW_LIB AND NOT BUILD_CPU_MKL_LIB)
        CONTINUE()
    ELSEIF((${TEST_NAME} MATCHES "Fftw") AND NOT BUILD_CPU_FFTW_LIB)
        CONTINUE()
    ELSEIF((${TEST_NAME} MATCHES "Mkl") AND NOT BUILD_CPU_MKL_LIB)
        CONTINUE()
    ENDIF()

    ADD_EXECUTABLE(${TEST_NAME} ${FILE_NAME})
    TARGET_LINK_LIBRARIES(${TEST_NAME} PRIVATE polymerfts_backends)

    IF(${TEST_NAME} MATCHES "Scft" OR ${TEST_NAME} MATCHES "Stress" OR ${TEST_NAME} MATCHES "Hexagonal")
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
    TARGET_LINK_LIBRARIES(${BENCH_NAME} PRIVATE polymerfts_backends)
ENDFOREACH()

IF(CUDAToolkit_FOUND)
    # Generate files named "Test*.cu"
    FILE(GLOB FILE_LIST "Test*.cu")
    FOREACH(FILE_NAME ${FILE_LIST})
        GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)

        # Several CUDA tests use FFTW as the CPU reference (#include <fftw3.h>
        # without a USE_CPU_FFTW guard); they cannot build without it.
        IF((${TEST_NAME} MATCHES "Fftw|CrysFFT|CudaDCT|CudaDST") AND NOT BUILD_CPU_FFTW_LIB)
            CONTINUE()
        ENDIF()

        ADD_EXECUTABLE(${TEST_NAME} ${FILE_NAME})
        TARGET_LINK_LIBRARIES(${TEST_NAME} PRIVATE polymerfts_backends)

        # Same architecture list as the cuda library
        SET_PROPERTY(TARGET ${TEST_NAME} PROPERTY CUDA_ARCHITECTURES ${POLYMERFTS_CUDA_ARCHITECTURES})
        # Benchmarks (Test*Bench*.cu) build as executables but are not tests.
        IF(NOT ${TEST_NAME} MATCHES "Bench")
            ADD_TEST(${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
            SET_TEST_LABELS(${TEST_NAME})
        ENDIF()
    ENDFOREACH()
    # Its out-of-memory check holds most of the free device memory for a moment
    # (a ballast buffer); keep it away from the other tests under ctest -j
    IF(TARGET TestCudaDeviceBuffer)
        SET_TESTS_PROPERTIES(TestCudaDeviceBuffer PROPERTIES RUN_SERIAL TRUE)
    ENDIF()
ENDIF()
