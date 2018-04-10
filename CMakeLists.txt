cmake_minimum_required (VERSION 3.0)

project (qmkl)

set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_CONTACT "Sugizaki Yukimasa <ysugi@idein.jp>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Math Kernel Library for VideoCore IV QPU")

# See https://semver.org/
set(CPACK_PACKAGE_VERSION_MAJOR 0)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 1)

set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
include(CPack)

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

pkg_check_modules(MAILBOX REQUIRED libmailbox>=2.0.0)

# librpimemmgr needs bcm_host and vcsm, which may be in /opt/vc...
pkg_check_modules(RPIMEMMGR librpimemmgr>=1.0.0)
if (NOT RPIMEMMGR_FOUND)
    message(STATUS "Adding /opt/vc/lib/pkgconfig to PKG_CONFIG_PATH")
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/opt/vc/lib/pkgconfig")
    pkg_check_modules(RPIMEMMGR librpimemmgr>=1.0.0)
    if (NOT RPIMEMMGR_FOUND)
        message (FATAL_ERROR "librpimemmgr not found even in /opt/vc/lib. "
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
    message(WARNING "RPIVER is not specified; using default: 1")
    set (RPIVER 1)
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

configure_file(qmkl.pc.in qmkl.pc @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qmkl.pc"
        DESTINATION lib/pkgconfig)
