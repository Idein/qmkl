add_library (blas OBJECT gemm.c)

qasm2m4_dep_on_c (gemm.c sgemm)
