# qbin_dep_on_c (c_filename qbin_basename1 qbin_basename2 ...)
macro (qbin_dep_on_c c_filename)

    foreach (basename ${ARGN})

        # Todo: We cannot use get_source_file_property
        #       because it appends '\' every after variables.
        set (${c_filename}_deps ${${c_filename}_deps} ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex)
        set_source_files_properties (
            ${c_filename} PROPERTIES
                OBJECT_DEPENDS "${${c_filename}_deps}"
        )

        add_custom_command (
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex
            COMMAND ${QBIN2HEX} <${CMAKE_CURRENT_SOURCE_DIR}/${basename}.qbin >${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${basename}.qbin
        )

    endforeach (basename)

endmacro (qbin_dep_on_c)
