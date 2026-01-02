ENABLE_TESTING()

# Add Python Files
FILE(GLOB FILE_LIST "Test*.py")
FOREACH(FILE_NAME ${FILE_LIST})
    GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)
    ADD_TEST(NAME ${TEST_NAME} COMMAND ${Python3_EXECUTABLE} ${FILE_NAME})
ENDFOREACH()

# Generate files named "Test*.cpp"
FILE(GLOB FILE_LIST "Test*.cpp")
FOREACH(FILE_NAME ${FILE_LIST})
    GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)
    IF ((${TEST_NAME} MATCHES "Cuda") AND NOT CUDAToolkit_FOUND)
        CONTINUE()
    ELSEIF((${TEST_NAME} MATCHES "Cpu") AND NOT BUILD_CPU_MKL_LIB)
        CONTINUE()
    ENDIF()

    ADD_EXECUTABLE(${TEST_NAME} ${FILE_NAME})
    TARGET_LINK_LIBRARIES(${TEST_NAME} PRIVATE
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
    
    IF(${TEST_NAME} MATCHES "ParamParser" OR ${TEST_NAME} MATCHES "Scft" OR ${TEST_NAME} MATCHES "Stress")
        ADD_TEST(NAME ${TEST_NAME}
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
            )
    ELSE()
        ADD_TEST(${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
    ENDIF()
ENDFOREACH()

IF(CUDAToolkit_FOUND)
    # Generate files named "Test*.cu"
    FILE(GLOB FILE_LIST "Test*.cu")
    FOREACH(FILE_NAME ${FILE_LIST})
        GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)

        ADD_EXECUTABLE(${TEST_NAME} ${FILE_NAME})
        TARGET_LINK_LIBRARIES(${TEST_NAME} PRIVATE
            factory cuda CUDA::cufft
            $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,cpu-mkl,>
            $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lmkl_intel_lp64,>
            $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lmkl_sequential,>
            $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lmkl_core,>
            $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-ldl,>
            $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lpthread,>
            $<IF:$<BOOL:${BUILD_CPU_MKL_LIB}>,-lm,>
            common)

        SET_PROPERTY(TARGET ${TEST_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
        ADD_TEST(${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
    ENDFOREACH()
ENDIF()
