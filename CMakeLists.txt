cmake_minimum_required (VERSION 3.0)

project (qmkl)

find_program (QASM2 qasm2)
if (NOT QASM2)
	message (FATAL_ERROR "qasm2 not found. Install it from https://github.com/Terminus-IMRC/qpu-assembler2")
endif ()

add_subdirectory (src)
add_subdirectory (test)
