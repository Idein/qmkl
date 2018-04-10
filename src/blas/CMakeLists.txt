include_directories (
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_library (
    blas
    OBJECT
        gemm.c
        copy.c
)

c_dep_on_qhex_from_py (gemm.c sgemm_RNN sgemm_RNT sgemm_RTN sgemm_RTT)
qasm2m4_dep_on_c (copy.c scopy)
