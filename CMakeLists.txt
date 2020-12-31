#---------------------------------------------------
# OpenMM Deepmd Plugin
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
SET(CMAKE_VERBOSE_MAKEFILE OFF)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# Include the deepmd-kit.
SET(DEEPMD_DIR "/usr/local/deepmd/" CACHE PATH "Where deepmd-kit is installed")
INCLUDE_DIRECTORIES("${DEEPMD_DIR}/include/deepmd")
LINK_DIRECTORIES("${DEEPMD_DIR}/lib")

# We need to know where TensorFlow is installed so we can access the headers and libraries.
SET(TENSORFLOW_DIR "/usr/local/tensorflow/r1.14" CACHE PATH "Where TensorFlow is installed")
INCLUDE_DIRECTORIES("${TENSORFLOW_DIR}/include")
INCLUDE_DIRECTORIES("${TENSORFLOW_DIR}/include/tensorflow/c")
LINK_DIRECTORIES("${TENSORFLOW_DIR}/lib")

# Specify the C++ version we are building for.
SET (CMAKE_CXX_STANDARD 11)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wl,--allow-multiple-definition  -Wl,--no-as-needed")

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE(WIN32)
        SET(CMAKE_INSTALL_PREFIX "/usr/local/openmm" CACHE PATH "Where to install the plugin" FORCE)
    ENDIF(WIN32)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# Put all the tests and libraries in a single output directory.
IF(NOT EXECUTABLE_OUTPUT_PATH)
  SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}
      CACHE INTERNAL "Single output directory for building all executables.")
ENDIF()
IF(NOT LIBRARY_OUTPUT_PATH)
  SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}
      CACHE INTERNAL "Single output directory for building all libraries.")
ENDIF()
SET(${PROJECT_NAME}_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(${PROJECT_NAME}_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(DEEPMD_PLUGIN_SOURCE_SUBDIRS openmmapi serialization)

# Set the library name
SET(DEEPMD_LIBRARY_NAME OpenMMDeepmd)
SET(SHARED_DEEPMD_TARGET ${DEEPMD_LIBRARY_NAME})

# These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal" ${DeePMD_INCLUDE_DIRS})

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# Collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${DEEPMD_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# Create the library.

ADD_LIBRARY(${SHARED_DEEPMD_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_DEEPMD_TARGET}
    PROPERTIES COMPILE_FLAGS "-DDEEPMD_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}"
    COMPILE_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
TARGET_LINK_LIBRARIES(${SHARED_DEEPMD_TARGET} OpenMM)
TARGET_LINK_LIBRARIES(${SHARED_DEEPMD_TARGET} deepmd)
TARGET_LINK_LIBRARIES(${SHARED_DEEPMD_TARGET} tensorflow_cc)
TARGET_LINK_LIBRARIES(${SHARED_DEEPMD_TARGET} tensorflow_framework)

INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_DEEPMD_TARGET})

# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

# Enable testing
# Set no testing here.
#NABLE_TESTING()
#ADD_SUBDIRECTORY(serialization/tests)

#[[
# Copy test files to the build directory.

file(GLOB_RECURSE TEST_FILES RELATIVE "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/tests/*.pb"
)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
set(COPIED_TEST_FILES)
foreach(TEST_FILE ${TEST_FILES})
    set(infile "${CMAKE_SOURCE_DIR}/${TEST_FILE}")
    set(outfile "${CMAKE_BINARY_DIR}/${TEST_FILE}")
    add_custom_command(
        OUTPUT "${outfile}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${infile}" "${outfile}"
        DEPENDS "${infile}"
        COMMENT "CMake-copying file ${infile} to ${outfile}")
    set(COPIED_TEST_FILES ${COPIED_TEST_FILES} "${outfile}")
endforeach()
add_custom_target(CopyTestFiles ALL DEPENDS ${COPIED_TEST_FILES})
]]#
# Build the implementations for different platforms

# Use reference platform only here.
ADD_SUBDIRECTORY(platforms/reference)


FIND_PACKAGE(CUDA QUIET)
IF(CUDA_FOUND)
    SET(NN_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for CUDA")
ELSE(CUDA_FOUND)
    SET(NN_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for CUDA")
ENDIF(CUDA_FOUND)
IF(NN_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(NN_BUILD_CUDA_LIB)

# Build the Python API

FIND_PROGRAM(PYTHON_EXECUTABLE python)
FIND_PROGRAM(SWIG_EXECUTABLE swig)
IF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(NN_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
ELSE(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(NN_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
ENDIF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
IF(NN_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(NN_BUILD_PYTHON_WRAPPERS)

