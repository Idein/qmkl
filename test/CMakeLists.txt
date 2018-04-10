set(QMKL_INCLUDE_DIRS ${VCSM_INCLUDE_DIRS} ${MAILBOX_INCLUDE_DIRS}
                      ${RPIMEMMGR_INCLUDE_DIRS})
set(QMKL_CFLAGS_OTHER ${VCSM_CFLAGS_OTHER} ${MAILBOX_CFLAGS_OTHER}
                      ${RPIMEMMGR_CFLAGS_OTHER})
set(QMKL_LDFLAGS -lm ${VCSM_LDFLAGS} ${MAILBOX_LDFLAGS} ${RPIMEMMGR_LDFLAGS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -O2 -g -W -Wall -Wextra \
                   ${OpenMP_C_FLAGS}")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${QMKL_INCLUDE_DIRS}
)

add_executable(sgemm sgemm.c)
target_compile_options(sgemm PRIVATE "${QMKL_CFLAGS_OTHER}")
target_link_libraries(sgemm qmkl "${QMKL_LDFLAGS}")

add_executable(scopy scopy.c)
target_compile_options(scopy PRIVATE "${QMKL_CFLAGS_OTHER}")
target_link_libraries(scopy qmkl "${QMKL_LDFLAGS}")

add_executable(vsAbs vsAbs.c)
target_compile_options(vsAbs PRIVATE "${QMKL_CFLAGS_OTHER}")
target_link_libraries(vsAbs qmkl "${QMKL_LDFLAGS}")

include(FindPNG)

if (PNG_FOUND)
    option(HAVE_PNG "have libpng" ON)
endif (PNG_FOUND)

include (../cmake/FindCUnit.cmake)

if (CUNIT_FOUND)
    add_executable(sgemm_spec sgemm_spec.c)
    add_dependencies(sgemm_spec qmkl-static)
    target_include_directories(sgemm_spec PRIVATE ${CUNIT_INCLUDE_DIRS})
    target_compile_options(sgemm_spec PRIVATE ${QMKL_CFLAGS_OTHER})
    target_link_libraries(sgemm_spec "${CMAKE_BINARY_DIR}/src/libqmkl.a"
                          ${CUNIT_LIBRARIES} ${PNG_LIBRARIES} ${QMKL_LDFLAGS})

    add_executable(memory_bench memory_bench.c)
    add_dependencies(memory_bench qmkl-static)
    target_include_directories(memory_bench PRIVATE "${CUNIT_INCLUDE_DIRS}")
    target_compile_options(memory_bench PRIVATE "${QMKL_CFLAGS_OTHER}")
    target_link_libraries(memory_bench "${CMAKE_BINARY_DIR}/src/libqmkl.a"
                          ${CUNIT_LIBRARIES} ${QMKL_LDFLAGS})
endif (CUNIT_FOUND)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.h)
