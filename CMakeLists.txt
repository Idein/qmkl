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
    message (FATAL_ERROR "Please add /opt/vc/lib/pkgconfig to PKG_CONFIG_PATH environment variable")
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
