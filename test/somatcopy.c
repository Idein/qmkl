#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>

static void cpu_somatcopy(
    char ordering, char trans,
    const size_t rows, const size_t cols,
    const float alpha,
    const float *A, const size_t lda,
    float *B, const size_t ldb)
{
    size_t i, j;

    ordering = tolower(ordering);
    trans = tolower(trans);

    if (ordering == 'r') {
        if (trans == 'n')
            for (i = 0; i < rows; i ++)
                for (j = 0; j < cols; j ++)
                    B[i * ldb + j] = alpha * A[i * lda + j];
        else if (trans == 't')
            for (i = 0; i < rows; i ++)
                for (j = 0; j < cols; j ++)
                    B[j * ldb + i] = alpha * A[i * lda + j];
    } else if (ordering == 'c') {
        if (trans == 'n')
            for (i = 0; i < cols; i ++)
                for (j = 0; j < rows; j ++)
                    B[i * ldb + j] = alpha * A[i * lda + j];
        else if (trans == 't')
            for (i = 0; i < cols; i ++)
                for (j = 0; j < rows; j ++)
                    B[j * ldb + i] = alpha * A[i * lda + j];
    }

}

static float rand_float_in_range(float from, float to)
{
    return ((float) rand() / RAND_MAX) * (to - from) + from;
}

static float* mkl_malloc_randoms(const int m, const int n)
{
    float *p = mkl_malloc(m * n * sizeof(float), 4096);
    int i = 0;
#pragma omp parallel for private(i)
    for (i = 0; i < m * n; i ++)
        p[i] = rand_float_in_range(-1.0, 1.0);
    return p;
}

int main()
{
    const int rows = 16 * 100, cols = 16 + 32 * 120;
    float *A = NULL, *B = NULL, *B_ref = NULL;
    struct timeval start, end;
    float time;

    printf("%f [MB] of memory per a matrix\n", rows * cols * sizeof(*A) * 1e-6);
    fflush(stdout);

    A     = mkl_malloc_randoms(rows, cols);
    B     = mkl_malloc_randoms(rows, cols);
    B_ref = mkl_malloc_randoms(rows, cols);

    gettimeofday(&start, NULL);
    cpu_somatcopy('r', 'n', rows, cols, 1.0, A, cols, B_ref, cols);
    gettimeofday(&end, NULL);
    time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
    printf("CPU: %f [s], %f [MB/s]\n", time, (rows * cols * sizeof(*A)) / time * 1e-6);
    fflush(stdout);

    gettimeofday(&start, NULL);
    mkl_somatcopy('r', 'n', rows, cols, 1.0, A, cols, B,     cols);
    gettimeofday(&end, NULL);
    time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
    printf("GPU: %f [s], %f [MB/s]\n", time, (rows * cols * sizeof(*A)) / time * 1e-6);
    fflush(stdout);

    if (memcmp(B_ref, B, rows * cols * sizeof(*A)))
        printf("** CPU and GPU results differ! **\n");

    mkl_free(B_ref);
    mkl_free(B);
    mkl_free(A);
    return 0;
}
