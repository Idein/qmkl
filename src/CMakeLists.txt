include (../cmake/qasm2m4_dep_on_c.cmake)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -O2 -g -Wall -Wextra")

include_directories (
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory (blas)

add_library (
	qmkl SHARED
	main.c
	mailbox.c
	memory.c
	launch_qpu_code.c
	error.c
	$<TARGET_OBJECTS:blas>
)

install (
	TARGETS qmkl LIBRARY
	DESTINATION lib
)
install (
	FILES
		include/qmkl.h
	DESTINATION include
)
install (
	FILES
		include/qmkl/types.h
		include/qmkl/mailbox.h
		include/qmkl/memory.h
		include/qmkl/launch_qpu_code.h
		include/qmkl/blas.h
	DESTINATION include/qmkl
)
