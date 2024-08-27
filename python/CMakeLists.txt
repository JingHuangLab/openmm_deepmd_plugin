set(WRAP_FILE OpenMMDeepmdPluginWrapper.cpp)
set(MODULE_NAME OpenMMDeepmdPlugin)

# Execute SWIG to generate source code for the Python module.
add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "-I{DEEPMD_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDeepmdPlugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDeepmdPlugin.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Set up the git version name
# Get the latest abbreviated commit hash of the working branch
execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

# Compile the Python module.
add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(DEEPMDPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(DEEPMDPLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDeepmdPlugin DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDeepmdPlugin/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/OpenMMDeepmdPlugin/__init__.py)
add_custom_command(TARGET PythonInstall
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/OpenMMDeepmdPlugin.py ${CMAKE_CURRENT_BINARY_DIR}/OpenMMDeepmdPlugin
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install . 
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

