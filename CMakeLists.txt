cmake_minimum_required (VERSION 3.0)

project (qmkl)

find_package(PkgConfig)
find_package(OpenMP)

find_program (QASM2 qasm2)
if (NOT QASM2)
    message (FATAL_ERROR "qasm2 not found. Install it from https://github.com/Terminus-IMRC/qpu-assembler2")
endif ()

find_program (QBIN2HEX qbin2hex)
if (NOT QBIN2HEX)
    message (FATAL_ERROR "qbin2hex not found. Install it from https://github.com/Terminus-IMRC/qpu-bin-to-hex")
endif ()

find_program (M4 m4)
if (NOT M4)
    message (FATAL_ERROR "m4 not found.")
endif ()

pkg_check_modules(VCSM vcsm)
if (NOT VCSM_FOUND)
    message(STATUS "Adding /opt/vc/lib/pkgconfig to PKG_CONFIG_PATH")
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/opt/vc/lib/pkgconfig")
    pkg_check_modules(VCSM vcsm)
    if (NOT VCSM_FOUND)
        message (FATAL_ERROR "vcsm not found even in /opt/vc/lib. "
                             "Building on non-RPi host? "
                             "Please specify PKG_CONFIG_PATH.")
    endif ()
endif ()

if (DEFINED ENV{RPIVER})
    if     ("$ENV{RPIVER}" STREQUAL "1")
        set (RPIVER 1)
    elseif ("$ENV{RPIVER}" STREQUAL "2")
        set (RPIVER 2)
    elseif ("$ENV{RPIVER}" STREQUAL "3")
        set (RPIVER 3)
    else ()
        message (FATAL_ERROR "Invalid RPIVER specified: $ENV{RPIVER}")
    endif ()
else ()
    execute_process (
        COMMAND uname -m
        OUTPUT_VARIABLE UNAME_M
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (UNAME_M STREQUAL "armv6l")
        set (RPIVER 1)
    else ()
        if (NOT EXISTS /proc/cpuinfo)
            message (FATAL_ERROR "/proc is not mounted")
        endif ()
        file (STRINGS /proc/cpuinfo IS_RPI3
                      REGEX "^Features[ \t]*:.*[ \t]*crc32[ \t]*$")
        if (IS_RPI3)
            set (RPIVER 3)
        else ()
            set (RPIVER 2)
        endif ()
        unset(IS_RPI3)
    endif ()
    unset(UNAME_M)
endif ()

if (RPIVER EQUAL 1)
    message (STATUS "Building for Raspberry Pi 1 / Zero")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=arm1176jzf-s \
                        -mtune=arm1176jzf-s -mfloat-abi=hard -mfpu=vfp")
elseif (RPIVER EQUAL 2)
    message (STATUS "Building for Raspberry Pi 2")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-a7 -mtune=cortex-a7 \
                        -mfloat-abi=hard -mfpu=neon-vfpv4")
elseif (RPIVER EQUAL 3)
    message (STATUS "Building for Raspberry Pi 3")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-a53 -mtune=cortex-a53 \
                        -mfloat-abi=hard -mfpu=neon-vfpv4")
endif ()

add_subdirectory (src)
add_subdirectory (test)

enable_testing()

include (cmake/FindCUnit.cmake)

if(CUNIT_FOUND)
add_test(sgemm_spec sudo ./test/sgemm_spec)
add_custom_target(
    check
    COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS sgemm_spec
)
endif(CUNIT_FOUND)
