# qasm2m4_dep_on_c (c_filename qasm2m4_basename1 qasm2m4_basename2 ...)
function (qasm2m4_dep_on_c c_filename)

    foreach (basename ${ARGN})

        SET_SOURCE_FILES_PROPERTIES (
            ${c_filename} PROPERTIES
                OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex
        )

        add_custom_command (
            OUTPUT ${basename}.qhex
            COMMAND ${QBIN2HEX} <${basename}.qbin >${basename}.qhex
            DEPENDS ${basename}.qbin
        )

        add_custom_command (
            OUTPUT ${basename}.qbin
            COMMAND ${QASM2} <${basename}.qasm2 >${basename}.qbin
            DEPENDS ${basename}.qasm2
        )

        add_custom_command (
            OUTPUT ${basename}.qasm2
            COMMAND ${M4} <${CMAKE_CURRENT_SOURCE_DIR}/${basename}.qasm2m4 >${basename}.qasm2
            DEPENDS ${basename}.qasm2m4
        )

    endforeach (basename)

endfunction (qasm2m4_dep_on_c)
