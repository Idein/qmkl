add_library (
    blas
    OBJECT
        gemm.c
        copy.c
)

qbin_dep_on_c (gemm.c sgemm_16x64_1thread sgemm_16x64 sgemm_16x16)
qasm2m4_dep_on_c (copy.c scopy)
