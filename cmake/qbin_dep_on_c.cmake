# qbin_dep_on_c (c_filename qbin_basename1 qbin_basename2 ...)
function (qbin_dep_on_c c_filename)

	foreach (basename ${ARGN})

		SET_SOURCE_FILES_PROPERTIES (
			${c_filename} PROPERTIES
				OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${basename}.qhex
		)

		add_custom_command (
			OUTPUT ${basename}.qhex
			COMMAND ${QBIN2HEX} <${CMAKE_CURRENT_SOURCE_DIR}/${basename}.qbin >${basename}.qhex
			DEPENDS ${basename}.qbin
		)

	endforeach (basename)

endfunction (qbin_dep_on_c)
