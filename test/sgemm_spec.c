#include "config.h"
#include <unistd.h>
#include <stdlib.h>
#ifdef HAVE_PNG
#include <png.h>
#endif
#include <sys/time.h>
#include <omp.h>
#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include "mkl.h"

static double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + t.tv_usec * 1e-6;
}

#ifdef HAVE_PNG
static void visualize(const char* file, const int h, const int w, float* p) {
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    FILE* fp = fopen(file, "w");
    png_init_io(png, fp);
    png_set_IHDR(png, info, w, h, 8,
                 PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_bytepp rows = png_malloc(png, sizeof(png_bytep) * h);
    png_set_rows(png, info, rows);
    int i, j;
    for (i = 0; i < h; ++i) rows[i] = png_malloc(png, w);
    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            rows[i][j] = (png_byte)p[i*w+j]; // p[*] : 0 ~ 255
        }
    }
    png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);
    for (i = 0; i < h; ++i) png_free(png, rows[i]);
    png_free(png, rows);
    png_destroy_write_struct(&png, &info);
}
#endif

static void suite_sgemm_RNN();
static void suite_sgemm_RNT();
static void suite_sgemm_RTN();
static void suite_sgemm_RTT();
static void suite_sgemm_with_mempool();

int main() {
    CU_initialize_registry();

    suite_sgemm_RNN();
    suite_sgemm_RNT();
    suite_sgemm_RTN();
    suite_sgemm_RTT();
    suite_sgemm_with_mempool();

    isatty(fileno(stdout)) ? CU_console_run_tests() : CU_basic_run_tests();
    const unsigned int result = CU_get_number_of_failures();
    CU_cleanup_registry();
    return (result ? 1 : 0);
}

#define DECL_TEST_FOR_EACH_SIZE(TEST_FUNCTION) \
    static void TEST_FUNCTION##_S();           \
    static void TEST_FUNCTION##_M();           \
    static void TEST_FUNCTION##_L();
DECL_TEST_FOR_EACH_SIZE(test_sgemm_RNN_ones);
DECL_TEST_FOR_EACH_SIZE(test_sgemm_RNN_randoms);
static void test_sgemm_RNN_benchmark();

int setup_suite_sgemm_RNN() {
    srand(0xDEADBEEF);
    return 0;
}

int teardown_suite_sgemm_RNN() {
    return 0;
}

void suite_sgemm_RNN() {
    CU_pSuite suite = CU_add_suite("sgemm RNN", setup_suite_sgemm_RNN, teardown_suite_sgemm_RNN);

    CU_add_test(suite, "ones (small)", test_sgemm_RNN_ones_S);
    CU_add_test(suite, "ones (medium)", test_sgemm_RNN_ones_M);
    CU_add_test(suite, "ones (large)", test_sgemm_RNN_ones_L);
    CU_add_test(suite, "randoms (small)", test_sgemm_RNN_randoms_S);
    CU_add_test(suite, "randoms (medium)", test_sgemm_RNN_randoms_M);
    CU_add_test(suite, "randoms (large)", test_sgemm_RNN_randoms_L);
    CU_add_test(suite, "benchmark", test_sgemm_RNN_benchmark);
}

static float* mkl_malloc_ones(const int m, const int n) {
    float* p = mkl_malloc(m * n * sizeof(float), 4096);
    int i = 0;
    for (i = 0; i < m * n; ++i) p[i] = 1.0;
    return p;
}

static void test_sgemm_RNN_ones(const int M, const int N, const int K) {
    float* A = mkl_malloc_ones(M, K);
    float* B = mkl_malloc_ones(K, N);
    float* C = mkl_malloc_ones(M, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 1, C, N);
    int diff = 0;
    {
        int i, j;
        for (i = 0; i < M; ++i)
            for (j = 0; j < N; ++j)
                diff |= K+1 - (int)C[i*N+j];
        CU_ASSERT_EQUAL(diff, 0);
    }
#ifdef HAVE_PNG
    if (diff) {
        {
            int i, j;
#pragma omp parallel for private(i, j)
            for (i = 0; i < M; ++i)
                for (j = 0; j < N; ++j)
                    C[i*N+j] = (K+1 == (int)C[i*N+j]) ? 255 : 0;
        }
        char file[256] = {0};
        sprintf(file, "ones_%dx%d_%dx%d.png", M, K, K, N);
        visualize(file, M, N, C);
    }
#endif
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

static int rand_int_in_range(int from, int to) {
    return ((float)rand() / RAND_MAX) * (to - from) + from;
}

#define IMPL_TEST_FOR_EACH_SIZE(TEST_FUNCTION)              \
    void TEST_FUNCTION##_S() {                              \
        int i = 0;                                          \
        puts("");                                           \
        for (i = 0; i < 32; ++i) {                          \
            int M = rand_int_in_range(1, 256);              \
            int N = rand_int_in_range(1, 256);              \
            int K = rand_int_in_range(2, 128);              \
            printf("M = %d, N = %d K = %d\n", M, N, K);     \
            TEST_FUNCTION(M, N, K);                         \
        }                                                   \
    }                                                       \
    void TEST_FUNCTION##_M() {                              \
        int i = 0;                                          \
        puts("");                                           \
        for (i = 0; i < 16; ++i) {                          \
            int M = rand_int_in_range(1, 64 * 12);          \
            int N = rand_int_in_range(1, 64 * 12);          \
            int K = rand_int_in_range(2, 128);              \
            printf("M = %d, N = %d K = %d\n", M, N, K);     \
            TEST_FUNCTION(M, N, K);                         \
        }                                                   \
    }                                                       \
    void TEST_FUNCTION##_L() {                              \
        int i = 0;                                          \
        puts("");                                           \
        for (i = 0; i < 8; ++i) {                           \
            int M = rand_int_in_range(64 * 12, 1024);       \
            int N = rand_int_in_range(4096, 8192);          \
            int K = rand_int_in_range(2, 128);              \
            printf("M = %d, N = %d K = %d\n", M, N, K);     \
            TEST_FUNCTION(M, N, K);                         \
        }                                                   \
    }

IMPL_TEST_FOR_EACH_SIZE(test_sgemm_RNN_ones);

static float rand_float_in_range(float from, float to) {
    return ((float)rand() / RAND_MAX) * (to - from) + from;
}

static float* mkl_malloc_randoms(const int m, const int n) {
    float* p = mkl_malloc(m * n * sizeof(float), 4096);
    int i = 0;
#pragma omp parallel for private(i)
    for (i = 0; i < m * n; ++i) p[i] = rand_float_in_range(-1.0, 1.0);
    return p;
}

static void test_sgemm_RNN_randoms(const int M, const int N, const int K) {
    float* A = mkl_malloc_randoms(M, K);
    float* B = mkl_malloc_randoms(K, N);
    float* C = mkl_malloc_randoms(M, N);
    float* A_ref = malloc(K*M*sizeof(float));
    float* B_ref = malloc(N*K*sizeof(float));
    float* C_ref = malloc(M*N*sizeof(float));
    memcpy(A_ref, A, M*K*sizeof(float));
    memcpy(B_ref, B, K*N*sizeof(float));
    memcpy(C_ref, C, M*N*sizeof(float));
    const float alpha = rand_float_in_range(-1.0, 1.0);
    const float beta = rand_float_in_range(-1.0, 1.0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
    {
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A_ref[i*K+k] * B_ref[k*N+j];
                C_ref[i*N+j] = alpha * acc + beta * C_ref[i*N+j];
            }
        }
    }
    {
        float maximum_abs_error = 0;
        int i, j;
#pragma omp parallel for private(i, j) reduction(max: maximum_abs_error)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                if (maximum_abs_error < fabsf(C_ref[i*N+j] - C[i*N+j]))
                    maximum_abs_error = fabsf(C_ref[i*N+j] - C[i*N+j]);
            }
        }
        CU_ASSERT_DOUBLE_EQUAL(maximum_abs_error, 0, 0.001);
    }
    free(C_ref);
    free(B_ref);
    free(A_ref);
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

IMPL_TEST_FOR_EACH_SIZE(test_sgemm_RNN_randoms);

void test_sgemm_RNN_benchmark() {
    const int M = 96;
    const int N = 3072;
    const int K = 363;
    float* A = mkl_malloc_randoms(M, K);
    float* B = mkl_malloc_randoms(K, N);
    float* C = mkl_malloc_randoms(M, N);
    float* A_ref = malloc(K*M*sizeof(float));
    float* B_ref = malloc(N*K*sizeof(float));
    float* C_ref = malloc(M*N*sizeof(float));
    memcpy(A_ref, A, M*K*sizeof(float));
    memcpy(B_ref, B, K*N*sizeof(float));
    memcpy(C_ref, C, M*N*sizeof(float));
    const float alpha = rand_float_in_range(-1.0, 1.0);
    const float beta = rand_float_in_range(-1.0, 1.0);
    printf("\nRNN: %dx%d * %dx%d\n", M, K, K, N);
    {
        double start = get_time();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
        double elapsed_time = get_time() - start;
        printf("GPU:             %9.6lf [sec], %9.6lf [Gflop/s]\n",
               elapsed_time, (2 * M * N * K + 3 * M * N) / elapsed_time * 1e-9);
    }
    {
        double start = get_time();
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A_ref[i*K+k] * B_ref[k*N+j];
                C_ref[i*N+j] = alpha * acc + beta * C_ref[i*N+j];
            }
        }
        double elapsed_time = get_time() - start;
        printf("CPU (%d threads): %9.6lf [sec], %9.6lf [Gflop/s]\n", omp_get_max_threads(),
               elapsed_time, (2 * M * N * K + 3 * M * N) / elapsed_time * 1e-9);
    }
    free(C_ref);
    free(B_ref);
    free(A_ref);
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

DECL_TEST_FOR_EACH_SIZE(test_sgemm_RNT_ones);
DECL_TEST_FOR_EACH_SIZE(test_sgemm_RNT_randoms);
static void test_sgemm_RNT_benchmark();

int setup_suite_sgemm_RNT() {
    srand(0xDEADBEEF);
    return 0;
}

int teardown_suite_sgemm_RNT() {
    return 0;
}

void suite_sgemm_RNT() {
    CU_pSuite suite = CU_add_suite("sgemm RNT", setup_suite_sgemm_RNT, teardown_suite_sgemm_RNT);

    CU_add_test(suite, "ones (small)", test_sgemm_RNT_ones_S);
    CU_add_test(suite, "ones (medium)", test_sgemm_RNT_ones_M);
    CU_add_test(suite, "ones (large)", test_sgemm_RNT_ones_L);
    CU_add_test(suite, "randoms (small)", test_sgemm_RNT_randoms_S);
    CU_add_test(suite, "randoms (medium)", test_sgemm_RNT_randoms_M);
    CU_add_test(suite, "randoms (large)", test_sgemm_RNT_randoms_L);
    CU_add_test(suite, "benchmark", test_sgemm_RNT_benchmark);
}

static void test_sgemm_RNT_ones(const int M, const int N, const int K) {
    float* A = mkl_malloc_ones(M, K);
    float* B = mkl_malloc_ones(N, K);
    float* C = mkl_malloc_ones(M, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1, A, K, B, K, 1, C, N);
    int diff = 0;
    {
        int i, j;
        for (i = 0; i < M; ++i)
            for (j = 0; j < N; ++j)
                diff |= K+1 - (int)C[i*N+j];
        CU_ASSERT_EQUAL(diff, 0);
    }
#ifdef HAVE_PNG
    if (diff) {
        {
            int i, j;
#pragma omp parallel for private(i, j)
            for (i = 0; i < M; ++i)
                for (j = 0; j < N; ++j)
                    C[i*N+j] = (K+1 == (int)C[i*N+j]) ? 255 : 0;
        }
        char file[256] = {0};
        sprintf(file, "ones_%dx%d_%dx%d.png", M, K, K, N);
        visualize(file, M, N, C);
    }
#endif
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

IMPL_TEST_FOR_EACH_SIZE(test_sgemm_RNT_ones);

static void test_sgemm_RNT_randoms(const int M, const int N, const int K) {
    float* A = mkl_malloc_randoms(M, K);
    float* B = mkl_malloc_randoms(N, K);
    float* C = mkl_malloc_randoms(M, N);
    float* A_ref = malloc(M*K*sizeof(float));
    float* B_ref = malloc(N*K*sizeof(float));
    float* C_ref = malloc(M*N*sizeof(float));
    memcpy(A_ref, A, M*K*sizeof(float));
    memcpy(B_ref, B, N*K*sizeof(float));
    memcpy(C_ref, C, M*N*sizeof(float));
    const float alpha = rand_float_in_range(-1.0, 1.0);
    const float beta = rand_float_in_range(-1.0, 1.0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, N);
    {
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A_ref[i*K+k] * B_ref[j*K+k];
                C_ref[i*N+j] = alpha * acc + beta * C_ref[i*N+j];
            }
        }
    }
    {
        float maximum_abs_error = 0;
        int i, j;
#pragma omp parallel for private(i, j) reduction(max: maximum_abs_error)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                if (maximum_abs_error < fabsf(C_ref[i*N+j] - C[i*N+j]))
                    maximum_abs_error = fabsf(C_ref[i*N+j] - C[i*N+j]);
            }
        }
        CU_ASSERT_DOUBLE_EQUAL(maximum_abs_error, 0, 0.001);
    }
    free(C_ref);
    free(B_ref);
    free(A_ref);
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

IMPL_TEST_FOR_EACH_SIZE(test_sgemm_RNT_randoms);

void test_sgemm_RNT_benchmark() {
    const int M = 96;
    const int N = 3072;
    const int K = 363;
    float* A = mkl_malloc_randoms(M, K);
    float* B = mkl_malloc_randoms(N, K);
    float* C = mkl_malloc_randoms(M, N);
    float* A_ref = malloc(M*K*sizeof(float));
    float* B_ref = malloc(N*K*sizeof(float));
    float* C_ref = malloc(M*N*sizeof(float));
    memcpy(A_ref, A, M*K*sizeof(float));
    memcpy(B_ref, B, N*K*sizeof(float));
    memcpy(C_ref, C, M*N*sizeof(float));
    const float alpha = rand_float_in_range(-1.0, 1.0);
    const float beta = rand_float_in_range(-1.0, 1.0);
    printf("\nRNT: %dx%d * %dx%d\n", M, K, K, N);
    {
        double start = get_time();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, N);
        double elapsed_time = get_time() - start;
        printf("GPU:             %9.6lf [sec], %9.6lf [Gflop/s]\n",
               elapsed_time, (2 * M * N * K + 3 * M * N) / elapsed_time * 1e-9);
    }
    {
        double start = get_time();
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A_ref[i*K+k] * B_ref[j*K+k];
                C_ref[i*N+j] = alpha * acc + beta * C_ref[i*N+j];
            }
        }
        double elapsed_time = get_time() - start;
        printf("CPU (%d threads): %9.6lf [sec], %9.6lf [Gflop/s]\n", omp_get_max_threads(),
               elapsed_time, (2 * M * N * K + 3 * M * N) / elapsed_time * 1e-9);
    }
    free(C_ref);
    free(B_ref);
    free(A_ref);
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

DECL_TEST_FOR_EACH_SIZE(test_sgemm_RTN_ones);
DECL_TEST_FOR_EACH_SIZE(test_sgemm_RTN_randoms);
static void test_sgemm_RTN_benchmark();

int setup_suite_sgemm_RTN() {
    srand(0xDEADBEEF);
    return 0;
}

int teardown_suite_sgemm_RTN() {
    return 0;
}

void suite_sgemm_RTN() {
    CU_pSuite suite = CU_add_suite("sgemm RTN", setup_suite_sgemm_RTN, teardown_suite_sgemm_RTN);

    CU_add_test(suite, "ones (small)", test_sgemm_RTN_ones_S);
    CU_add_test(suite, "ones (medium)", test_sgemm_RTN_ones_M);
    CU_add_test(suite, "ones (large)", test_sgemm_RTN_ones_L);
    CU_add_test(suite, "randoms (small)", test_sgemm_RTN_randoms_S);
    CU_add_test(suite, "randoms (medium)", test_sgemm_RTN_randoms_M);
    CU_add_test(suite, "randoms (large)", test_sgemm_RTN_randoms_L);
    CU_add_test(suite, "benchmark", test_sgemm_RTN_benchmark);
}

static void test_sgemm_RTN_ones(const int M, const int N, const int K) {
    float* A = mkl_malloc_ones(K, M);
    float* B = mkl_malloc_ones(K, N);
    float* C = mkl_malloc_ones(M, N);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1, A, M, B, N, 1, C, N);
    int diff = 0;
    {
        int i, j;
        for (i = 0; i < M; ++i)
            for (j = 0; j < N; ++j)
                diff |= K+1 - (int)C[i*N+j];
        CU_ASSERT_EQUAL(diff, 0);
    }
#ifdef HAVE_PNG
    if (diff) {
        {
            int i, j;
#pragma omp parallel for private(i, j)
            for (i = 0; i < M; ++i)
                for (j = 0; j < N; ++j)
                    C[i*N+j] = (K+1 == (int)C[i*N+j]) ? 255 : 0;
        }
        char file[256] = {0};
        sprintf(file, "ones_%dx%d_%dx%d.png", M, K, K, N);
        visualize(file, M, N, C);
    }
#endif
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

IMPL_TEST_FOR_EACH_SIZE(test_sgemm_RTN_ones);

static void test_sgemm_RTN_randoms(const int M, const int N, const int K) {
    float* A = mkl_malloc_randoms(K, M);
    float* B = mkl_malloc_randoms(K, N);
    float* C = mkl_malloc_randoms(M, N);
    float* A_ref = malloc(K*M*sizeof(float));
    float* B_ref = malloc(K*N*sizeof(float));
    float* C_ref = malloc(M*N*sizeof(float));
    memcpy(A_ref, A, K*M*sizeof(float));
    memcpy(B_ref, B, K*N*sizeof(float));
    memcpy(C_ref, C, M*N*sizeof(float));
    const float alpha = rand_float_in_range(-1.0, 1.0);
    const float beta = rand_float_in_range(-1.0, 1.0);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
    {
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A_ref[k*M+i] * B_ref[k*N+j];
                C_ref[i*N+j] = alpha * acc + beta * C_ref[i*N+j];
            }
        }
    }
    {
        float maximum_abs_error = 0;
        int i, j;
#pragma omp parallel for private(i, j) reduction(max: maximum_abs_error)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                if (maximum_abs_error < fabsf(C_ref[i*N+j] - C[i*N+j]))
                    maximum_abs_error = fabsf(C_ref[i*N+j] - C[i*N+j]);
            }
        }
        CU_ASSERT_DOUBLE_EQUAL(maximum_abs_error, 0, 0.001);
    }
    free(C_ref);
    free(B_ref);
    free(A_ref);
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

IMPL_TEST_FOR_EACH_SIZE(test_sgemm_RTN_randoms);

void test_sgemm_RTN_benchmark() {
    const int M = 96;
    const int N = 3072;
    const int K = 363;
    float* A = mkl_malloc_randoms(K, M);
    float* B = mkl_malloc_randoms(K, N);
    float* C = mkl_malloc_randoms(M, N);
    float* A_ref = malloc(K*M*sizeof(float));
    float* B_ref = malloc(K*N*sizeof(float));
    float* C_ref = malloc(M*N*sizeof(float));
    memcpy(A_ref, A, K*M*sizeof(float));
    memcpy(B_ref, B, K*N*sizeof(float));
    memcpy(C_ref, C, M*N*sizeof(float));
    const float alpha = rand_float_in_range(-1.0, 1.0);
    const float beta = rand_float_in_range(-1.0, 1.0);
    printf("\nRTN: %dx%d * %dx%d\n", M, K, K, N);
    {
        double start = get_time();
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
        double elapsed_time = get_time() - start;
        printf("GPU:             %9.6lf [sec], %9.6lf [Gflop/s]\n",
               elapsed_time, (2 * M * N * K + 3 * M * N) / elapsed_time * 1e-9);
    }
    {
        double start = get_time();
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A_ref[k*M+i] * B_ref[k*N+j];
                C_ref[i*N+j] = alpha * acc + beta * C_ref[i*N+j];
            }
        }
        double elapsed_time = get_time() - start;
        printf("CPU (%d threads): %9.6lf [sec], %9.6lf [Gflop/s]\n", omp_get_max_threads(),
               elapsed_time, (2 * M * N * K + 3 * M * N) / elapsed_time * 1e-9);
    }
    free(C_ref);
    free(B_ref);
    free(A_ref);
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

DECL_TEST_FOR_EACH_SIZE(test_sgemm_RTT_ones);
DECL_TEST_FOR_EACH_SIZE(test_sgemm_RTT_randoms);
static void test_sgemm_RTT_benchmark();

int setup_suite_sgemm_RTT() {
    srand(0xDEADBEEF);
    return 0;
}

int teardown_suite_sgemm_RTT() {
    return 0;
}

void suite_sgemm_RTT() {
    CU_pSuite suite = CU_add_suite("sgemm RTT", setup_suite_sgemm_RTT, teardown_suite_sgemm_RTT);

    CU_add_test(suite, "ones (small)", test_sgemm_RTT_ones_S);
    CU_add_test(suite, "ones (medium)", test_sgemm_RTT_ones_M);
    CU_add_test(suite, "ones (large)", test_sgemm_RTT_ones_L);
    CU_add_test(suite, "randoms (small)", test_sgemm_RTT_randoms_S);
    CU_add_test(suite, "randoms (medium)", test_sgemm_RTT_randoms_M);
    CU_add_test(suite, "randoms (large)", test_sgemm_RTT_randoms_L);
    CU_add_test(suite, "benchmark", test_sgemm_RTT_benchmark);
}

static void test_sgemm_RTT_ones(const int M, const int N, const int K) {
    float* A = mkl_malloc_ones(K, M);
    float* B = mkl_malloc_ones(N, K);
    float* C = mkl_malloc_ones(M, N);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, 1, A, M, B, K, 1, C, N);
    int diff = 0;
    {
        int i, j;
        for (i = 0; i < M; ++i)
            for (j = 0; j < N; ++j)
                diff |= K+1 - (int)C[i*N+j];
        CU_ASSERT_EQUAL(diff, 0);
    }
#ifdef HAVE_PNG
    if (diff) {
        {
            int i, j;
#pragma omp parallel for private(i, j)
            for (i = 0; i < M; ++i)
                for (j = 0; j < N; ++j)
                    C[i*N+j] = (K+1 == (int)C[i*N+j]) ? 255 : 0;
        }
        char file[256] = {0};
        sprintf(file, "ones_%dx%d_%dx%d.png", M, K, K, N);
        visualize(file, M, N, C);
    }
#endif
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

IMPL_TEST_FOR_EACH_SIZE(test_sgemm_RTT_ones);

static void test_sgemm_RTT_randoms(const int M, const int N, const int K) {
    float* A = mkl_malloc_randoms(K, M);
    float* B = mkl_malloc_randoms(N, K);
    float* C = mkl_malloc_randoms(M, N);
    float* A_ref = malloc(K*M*sizeof(float));
    float* B_ref = malloc(N*K*sizeof(float));
    float* C_ref = malloc(M*N*sizeof(float));
    memcpy(A_ref, A, K*M*sizeof(float));
    memcpy(B_ref, B, N*K*sizeof(float));
    memcpy(C_ref, C, M*N*sizeof(float));
    const float alpha = rand_float_in_range(-1.0, 1.0);
    const float beta = rand_float_in_range(-1.0, 1.0);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A, M, B, K, beta, C, N);
    {
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A_ref[k*M+i] * B_ref[j*K+k];
                C_ref[i*N+j] = alpha * acc + beta * C_ref[i*N+j];
            }
        }
    }
    {
        float maximum_abs_error = 0;
        int i, j;
#pragma omp parallel for private(i, j) reduction(max: maximum_abs_error)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                if (maximum_abs_error < fabsf(C_ref[i*N+j] - C[i*N+j]))
                    maximum_abs_error = fabsf(C_ref[i*N+j] - C[i*N+j]);
            }
        }
        CU_ASSERT_DOUBLE_EQUAL(maximum_abs_error, 0, 0.001);
    }
    free(C_ref);
    free(B_ref);
    free(A_ref);
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

IMPL_TEST_FOR_EACH_SIZE(test_sgemm_RTT_randoms);

void test_sgemm_RTT_benchmark() {
    const int M = 3072;
    const int N = 96;
    const int K = 363;
    float* A = mkl_malloc_randoms(K, M);
    float* B = mkl_malloc_randoms(N, K);
    float* C = mkl_malloc_randoms(M, N);
    float* A_ref = malloc(K*M*sizeof(float));
    float* B_ref = malloc(N*K*sizeof(float));
    float* C_ref = malloc(M*N*sizeof(float));
    memcpy(A_ref, A, K*M*sizeof(float));
    memcpy(B_ref, B, N*K*sizeof(float));
    memcpy(C_ref, C, M*N*sizeof(float));
    const float alpha = rand_float_in_range(-1.0, 1.0);
    const float beta = rand_float_in_range(-1.0, 1.0);
    printf("\nRTT: %dx%d * %dx%d\n", M, K, K, N);
    {
        double start = get_time();
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A, M, B, K, beta, C, N);
        double elapsed_time = get_time() - start;
        printf("GPU:             %9.6lf [sec], %9.6lf [Gflop/s]\n",
               elapsed_time, (2 * M * N * K + 3 * M * N) / elapsed_time * 1e-9);
    }
    {
        double start = get_time();
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A_ref[k*M+i] * B_ref[j*K+k];
                C_ref[i*N+j] = alpha * acc + beta * C_ref[i*N+j];
            }
        }
        double elapsed_time = get_time() - start;
        printf("CPU (%d threads): %9.6lf [sec], %9.6lf [Gflop/s]\n", omp_get_max_threads(),
               elapsed_time, (2 * M * N * K + 3 * M * N) / elapsed_time * 1e-9);
    }
    free(C_ref);
    free(B_ref);
    free(A_ref);
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

static void test_sgemm_with_mempool_ones();
static void test_sgemm_with_mempool_randoms();

int setup_suite_sgemm_with_mempool() {
    srand(0xDEADBEEF);
    return 0;
}

int teardown_suite_sgemm_with_mempool() {
    return 0;
}

void suite_sgemm_with_mempool() {
    CU_pSuite suite = CU_add_suite("sgemm with mempool", setup_suite_sgemm_with_mempool, teardown_suite_sgemm_with_mempool);

    CU_add_test(suite, "ones", test_sgemm_with_mempool_ones);
    CU_add_test(suite, "randoms", test_sgemm_with_mempool_randoms);
}

static void test_sgemm_with_mempool_ones() {
    const int M = 96;
    const int N = 3072;
    const int K = 363;
    const int size = M * K + K * N + M * N;
    float* pool = mkl_malloc(size * sizeof(float), 4096);
    {
        int i = 0;
        for (i = 0; i < size; ++i) {
            pool[i] = 1.0;
        }
    }
    float* A = pool;
    float* B = A + M * K;
    float* C = B + K * N;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 1, C, N);
    int diff = 0;
    {
        int i, j;
        for (i = 0; i < M; ++i)
            for (j = 0; j < N; ++j)
                diff |= K+1 - (int)C[i*N+j];
        CU_ASSERT_EQUAL(diff, 0);
    }
#ifdef HAVE_PNG
    if (diff) {
        {
            int i, j;
#pragma omp parallel for private(i, j)
            for (i = 0; i < M; ++i)
                for (j = 0; j < N; ++j)
                    C[i*N+j] = (K+1 == (int)C[i*N+j]) ? 255 : 0;
        }
        char file[256] = {0};
        sprintf(file, "ones_%dx%d_%dx%d.png", M, K, K, N);
        visualize(file, M, N, C);
    }
#endif
    mkl_free(pool);
}

void test_sgemm_with_mempool_randoms() {
    const int M = 96;
    const int N = 3072;
    const int K = 363;
    const int size = M * K + K * N + M * N;
    float* pool = mkl_malloc(size * sizeof(float), 4096);
    {
        int i = 0;
        for (i = 0; i < size; ++i) {
            pool[i] = rand_float_in_range(-1.0, 1.0);
        }
    }
    float* A = pool;
    float* B = A + M * K;
    float* C = B + K * N;
    float* A_ref = malloc(M*K*sizeof(float));
    float* B_ref = malloc(K*N*sizeof(float));
    float* C_ref = malloc(M*N*sizeof(float));
    memcpy(A_ref, A, M*K*sizeof(float));
    memcpy(B_ref, B, K*N*sizeof(float));
    memcpy(C_ref, C, M*N*sizeof(float));
    const float alpha = rand_float_in_range(-1.0, 1.0);
    const float beta = rand_float_in_range(-1.0, 1.0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
    {
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A_ref[i*K+k] * B_ref[k*N+j];
                C_ref[i*N+j] = alpha * acc + beta * C_ref[i*N+j];
            }
        }
    }
    {   // C = alpha * A * B + beta * C
        float maximum_abs_error = 0;
        int i, j;
#pragma omp parallel for private(i, j) reduction(max: maximum_abs_error)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                if (maximum_abs_error < fabsf(C_ref[i*N+j] - C[i*N+j]))
                    maximum_abs_error = fabsf(C_ref[i*N+j] - C[i*N+j]);
            }
        }
        CU_ASSERT_DOUBLE_EQUAL(maximum_abs_error, 0, 0.001);
    }
    {   // A and B are not modified
        int eq = 1;
        int i, j;
        for (i = 0; i < M; ++i) {
            for (j = 0; j < K; ++j) {
                eq &= A_ref[i*K+j] == A[i*K+j];
            }
        }
        for (i = 0; i < K; ++i) {
            for (j = 0; j < N; ++j) {
                eq &= B_ref[i*N+j] == B[i*N+j];
            }
        }
        CU_ASSERT(eq);
    }
    free(C_ref);
    free(B_ref);
    free(A_ref);
    mkl_free(pool);
}
