include (../cmake/qasm2m4_dep_on_c.cmake)
include (../cmake/qbin_dep_on_c.cmake)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -O2 -g -Wall -Wextra ${VCSM_CFLAGS}")

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory (blas)
add_subdirectory (vm)

list (APPEND qmkl_SOURCES
    main.c
    mailbox.c
    memory.c
    launch_qpu_code.c
    error.c
    $<TARGET_OBJECTS:blas>
    $<TARGET_OBJECTS:vm>
)

add_library (
    qmkl SHARED
    ${qmkl_SOURCES}
)

add_library(
    qmkl-static STATIC
    ${qmkl_SOURCES}
)

set_target_properties(
    qmkl-static PROPERTIES
    OUTPUT_NAME "qmkl"
)

install (
    TARGETS qmkl LIBRARY
    DESTINATION lib
)
install (
    TARGETS qmkl-static LIBRARY
    DESTINATION lib
    ARCHIVE DESTINATION lib
)
install (
    FILES
        include/mkl.h
        include/qmkl.h
    DESTINATION include
)
install (
    FILES
        include/qmkl/types.h
        include/qmkl/mailbox.h
        include/qmkl/raspberrypi-firmware.h
        include/qmkl/memory.h
        include/qmkl/launch_qpu_code.h
        include/qmkl/blas.h
        include/qmkl/vm.h
        include/qmkl/error.h
    DESTINATION include/qmkl
)
