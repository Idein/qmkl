set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -O2 -g -Wall -Wextra")

include_directories (
	${CMAKE_CURRENT_SOURCE_DIR}/../src/include
)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
add_executable (
	sgemm
	sgemm.c
)
target_link_libraries (
	sgemm LINK_PUBLIC
	qmkl
)
