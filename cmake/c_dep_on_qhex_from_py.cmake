# c_dep_on_qhex_from_py(c_filename py_basename1 py_basename2 ...)
function (c_dep_on_qhex_from_py c_filename)

    foreach (basename ${ARGN})

        get_source_file_property(deps "${c_filename}" OBJECT_DEPENDS)
        if (NOT deps)
            unset(deps)
        endif ()
        set(deps ${deps} "${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex")
        set_source_files_properties(
            "${c_filename}" PROPERTIES OBJECT_DEPENDS "${deps}"
        )

        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex"
            COMMAND "${PYTHON_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.py" qhex
                            >"${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.py"
        )

    endforeach (basename)

endfunction (c_dep_on_qhex_from_py)
