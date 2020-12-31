set(WRAP_FILE OpenMMDeepmdPluginWrapper.cpp)
set(MODULE_NAME OpenMMDeepmdPlugin)

# Execute SWIG to generate source code for the Python module.

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "-I{TENSORFLOW_DIR}/include"
        "-I{DEEPMD_DIR}/include/deepmd"
        "${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDeepmdPlugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDeepmdPlugin.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.
add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(DEEPMDPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(DEEPMDPLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" setup.py build
    COMMAND "${PYTHON_EXECUTABLE}" setup.py install
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

