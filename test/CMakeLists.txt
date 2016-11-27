set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -O2 -g -Wall -Wextra")

include_directories (
	${CMAKE_CURRENT_SOURCE_DIR}/../src/include
)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

execute_process (
	COMMAND uname -m
	OUTPUT_VARIABLE UNAME_M
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
# If Raspberry Pi 2 or 3...
if (${UNAME_M} STREQUAL "armv7l")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfloat-abi=hard -mfpu=neon-vfpv4 -march=armv7-a -D_HAVE_NEON_")
endif ()

add_executable (
	sgemm
	sgemm.c
)
target_link_libraries (
	sgemm LINK_PUBLIC
	qmkl
)

add_executable (
	scopy
	scopy.c
)
target_link_libraries (
	scopy LINK_PUBLIC
	qmkl
)
