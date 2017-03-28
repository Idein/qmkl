# qasm2m4_dep_on_c (c_filename qasm2m4_basename1 qasm2m4_basename2 ...)
function (qasm2m4_dep_on_c c_filename)

    foreach (basename ${ARGN})

        SET_SOURCE_FILES_PROPERTIES (
            ${c_filename} PROPERTIES
                OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex
        )

        add_custom_command (
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex
            COMMAND ${QBIN2HEX} <${CMAKE_CURRENT_BINARY_DIR}/${basename}.qbin >${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qbin
        )

        add_custom_command (
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qbin
            COMMAND ${QASM2} <${CMAKE_CURRENT_BINARY_DIR}/${basename}.qasm2 >${CMAKE_CURRENT_BINARY_DIR}/${basename}.qbin
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qasm2
        )

        add_custom_command (
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qasm2
            COMMAND ${M4} <${CMAKE_CURRENT_SOURCE_DIR}/${basename}.qasm2m4 >${CMAKE_CURRENT_BINARY_DIR}/${basename}.qasm2
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${basename}.qasm2m4
        )

    endforeach (basename)

endfunction (qasm2m4_dep_on_c)
