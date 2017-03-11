add_library (
    blas
    OBJECT
        gemm.c
        copy.c
        omatcopy.c
)

qbin_dep_on_c (gemm.c sgemm_RNN sgemm_RNT sgemm_RTN sgemm_RTT)
qasm2m4_dep_on_c (copy.c scopy)
qasm2m4_dep_on_c (omatcopy.c somatcopy)
