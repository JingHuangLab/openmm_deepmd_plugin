CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
SET(CMAKE_VERBOSE_MAKEFILE ON)


SET(OPENMM_DIR "/usr/local/openmm_gcc6.5_7.5_cuda_9.2" CACHE PATH "Where OpenMM is installed")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# We need to know where TensorFlow is installed so we can access the headers and libraries.
SET(TENSORFLOW_ROOT "/home/dingye/.local/tensorflow/r1.14" CACHE PATH "Where TensorFlow is installed")
set(OP_CXX_ABI 1)
set (OP_CXX_FLAG -D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-ignored-attributes -Wl,--allow-multiple-definition -Wl,--whole-archive  -Wl,--no-as-needed")
set(TensorFlowFramework_LIBRARY "${TENSORFLOW_ROOT}/lib/libtensorflow_framework.so")
set(TensorFlowCore_LIBRARY "${TENSORFLOW_ROOT}/lib/libtensorflow_cc.so")
set (TensorFlowFramework_INCLUDE "${TENSORFLOW_ROOT}/include")
INCLUDE_DIRECTORIES("${TENSORFLOW_ROOT}/include")
INCLUDE_DIRECTORIES("${TENSORFLOW_ROOT}/include/tensorflow/c")

# Include the deepmd-kit.
SET(DEEPMD_DIR "/home/dingye/.local/deepmd-kit-1.2.0/" CACHE PATH "Where deepmd-kit is installed")
INCLUDE_DIRECTORIES("${DEEPMD_DIR}/include/deepmd")
LINK_DIRECTORIES("${DEEPMD_DIR}/lib")

# set prec
if (DEFINED FLOAT_PREC)
  string ( TOLOWER ${FLOAT_PREC} lower_float_prec )
  if (lower_float_prec STREQUAL "high")
    set(PREC_DEF "-DHIGH_PREC")
  else ()
    set(PREC_DEF "")    
  endif ()
else ()
    message("HIGH_PREC is used.")
    set(PREC_DEF "-DHIGH_PREC")
endif()
add_definitions (${PREC_DEF})

add_executable(testDeepmdPluginCUDA TestDeepmdPlugin4CUDA.cpp)
add_executable(testDeepmdPluginWithMicromd TestDeepmdPluginWithMicromd.cpp micromd.cpp)

target_link_libraries(testDeepmdPluginCUDA ${TensorFlowFramework_LIBRARY} ${TensorFlowCore_LIBRARY})
TARGET_LINK_LIBRARIES(testDeepmdPluginCUDA deepmd)
TARGET_LINK_LIBRARIES(testDeepmdPluginCUDA OpenMM)
TARGET_LINK_LIBRARIES(testDeepmdPluginCUDA OpenMMDeepmd)
TARGET_LINK_LIBRARIES(testDeepmdPluginCUDA OpenMMDeepmdCUDA)
TARGET_LINK_LIBRARIES(testDeepmdPluginCUDA OpenMMCUDA)
target_include_directories(testDeepmdPluginCUDA PUBLIC ${TensorFlowFramework_INCLUDE} ${PROJECT_BINARY_DIR})

target_link_libraries(testDeepmdPluginWithMicromd ${TensorFlowFramework_LIBRARY} ${TensorFlowCore_LIBRARY})
TARGET_LINK_LIBRARIES(testDeepmdPluginWithMicromd deepmd)
TARGET_LINK_LIBRARIES(testDeepmdPluginWithMicromd OpenMM)
TARGET_LINK_LIBRARIES(testDeepmdPluginWithMicromd OpenMMDeepmd)
TARGET_LINK_LIBRARIES(testDeepmdPluginWithMicromd OpenMMDeepmdCUDA)
TARGET_LINK_LIBRARIES(testDeepmdPluginWithMicromd OpenMMCUDA)
target_include_directories(testDeepmdPluginWithMicromd PUBLIC ${TensorFlowFramework_INCLUDE} ${PROJECT_BINARY_DIR})



set_target_properties(
    testDeepmdPluginCUDA
    PROPERTIES 
    COMPILE_FLAGS ${OP_CXX_FLAG}
    )

set_target_properties(
    testDeepmdPluginWithMicromd
    PROPERTIES 
    COMPILE_FLAGS ${OP_CXX_FLAG}
    )