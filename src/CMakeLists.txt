add_library (
	qmkl SHARED
	main.c
	mailbox.c
	memory.c
	error.c
)

include_directories (
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -O2 -g -Wall -Wextra")

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
	DESTINATION include/qmkl
)
