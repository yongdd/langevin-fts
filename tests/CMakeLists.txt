ENABLE_TESTING()

# Copy test data files to build directory
file(GLOB TEST_DATA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Stress*.txt")
file(COPY ${TEST_DATA_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Quick tests for installation verification (< 1 second each)
# Run with: ctest -L quick
SET(QUICK_TESTS
    # Python FFT tests
    TestNumPyFFT3D
    # CPU FFT tests
    TestCpuFFT3D
    # Basic propagator tests
    TestPseudoLinearContinuous3D
    TestPseudoLinearDiscrete3D
    # SCFT test
    TestScft1D
    # Platform verification
    TestDisplayArchitecture
    # CUDA tests (if available)
    TestCudaFFT
    TestCudaSimulationBox
)

# Helper function to set test labels
FUNCTION(SET_TEST_LABELS TEST_NAME)
    IF(${TEST_NAME} IN_LIST QUICK_TESTS)
        SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES LABELS "quick")
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
            common)

        SET_PROPERTY(TARGET ${TEST_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
        ADD_TEST(${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
        SET_TEST_LABELS(${TEST_NAME})
    ENDFOREACH()
ENDIF()
