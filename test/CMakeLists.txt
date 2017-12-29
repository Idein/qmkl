set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -O2 -g -Wall -Wextra")

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/include
    ${CMAKE_CURRENT_BINARY_DIR}
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
    m qmkl ${VCSM_LDFLAGS}
)

add_executable (
    scopy
    scopy.c
)
target_link_libraries (
    scopy LINK_PUBLIC
    qmkl ${VCSM_LDFLAGS}
)

add_executable (
    vsAbs
    vsAbs.c
)
target_link_libraries (
    vsAbs LINK_PUBLIC
    qmkl ${VCSM_LDFLAGS}
)


include(FindPNG)

if(PNG_FOUND)
option (HAVE_PNG  "have libpng" ON)
endif(PNG_FOUND)

include (../cmake/FindCUnit.cmake)

if(CUNIT_FOUND)
add_executable (
    sgemm_spec
    sgemm_spec.c
)
add_dependencies(sgemm_spec qmkl-static)
target_link_libraries (
    sgemm_spec
    LINK_PUBLIC ${CMAKE_BINARY_DIR}/src/libqmkl.a ${CUNIT_LIBRARIES} ${PNG_LIBRARIES} ${VCSM_LDFLAGS}
)
add_executable (
    memory_bench
    memory_bench.c
)
add_dependencies(memory_bench qmkl-static)
target_link_libraries (
    memory_bench
    LINK_PUBLIC ${CMAKE_BINARY_DIR}/src/libqmkl.a ${CUNIT_LIBRARIES} ${VCSM_LDFLAGS}
)
endif(CUNIT_FOUND)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/config.h)
