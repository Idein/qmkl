#include <unistd.h>
#include <stdlib.h>
#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include "mkl.h"

static void suite_sgemm_RNN();

int main() {
    CU_initialize_registry();

    suite_sgemm_RNN();

    isatty(fileno(stdout)) ? CU_console_run_tests() : CU_basic_run_tests();
    const unsigned int result = CU_get_number_of_failures();
    CU_cleanup_registry();
    return (result ? 1 : 0);
}

static void test_sgemm_RNN_ones_16x2_2x64();
static void test_sgemm_RNN_ones_16x2_2x128();
static void test_sgemm_RNN_ones_16x2_2x192();
static void test_sgemm_RNN_ones_16x2_2x256();
static void test_sgemm_RNN_ones_16x2_2x320();
static void test_sgemm_RNN_ones_16x2_2x384();
static void test_sgemm_RNN_ones_16x2_2x448();
static void test_sgemm_RNN_ones_32x2_2x64 ();
static void test_sgemm_RNN_ones_32x2_2x128();
static void test_sgemm_RNN_ones_32x2_2x192();
static void test_sgemm_RNN_ones_32x2_2x256();
static void test_sgemm_RNN_ones_32x2_2x320();
static void test_sgemm_RNN_ones_32x2_2x384();
static void test_sgemm_RNN_ones_32x2_2x448();
static void test_sgemm_RNN_ones_48x2_2x64 ();
static void test_sgemm_RNN_ones_48x2_2x128();
static void test_sgemm_RNN_ones_48x2_2x192();
static void test_sgemm_RNN_ones_48x2_2x256();
static void test_sgemm_RNN_ones_48x2_2x320();
static void test_sgemm_RNN_ones_48x2_2x384();
static void test_sgemm_RNN_ones_48x2_2x448();
static void test_sgemm_RNN_ones_96x363_363x3072();

static void test_sgemm_RNN_randoms_16x2_2x64();
static void test_sgemm_RNN_randoms_16x2_2x128();
static void test_sgemm_RNN_randoms_16x2_2x192();
static void test_sgemm_RNN_randoms_16x2_2x256();
static void test_sgemm_RNN_randoms_16x2_2x320();
static void test_sgemm_RNN_randoms_16x2_2x384();
static void test_sgemm_RNN_randoms_16x2_2x448();
static void test_sgemm_RNN_randoms_32x2_2x64 ();
static void test_sgemm_RNN_randoms_32x2_2x128();
static void test_sgemm_RNN_randoms_32x2_2x192();
static void test_sgemm_RNN_randoms_32x2_2x256();
static void test_sgemm_RNN_randoms_32x2_2x320();
static void test_sgemm_RNN_randoms_32x2_2x384();
static void test_sgemm_RNN_randoms_32x2_2x448();
static void test_sgemm_RNN_randoms_48x2_2x64 ();
static void test_sgemm_RNN_randoms_48x2_2x128();
static void test_sgemm_RNN_randoms_48x2_2x192();
static void test_sgemm_RNN_randoms_48x2_2x256();
static void test_sgemm_RNN_randoms_48x2_2x320();
static void test_sgemm_RNN_randoms_48x2_2x384();
static void test_sgemm_RNN_randoms_48x2_2x448();
static void test_sgemm_RNN_randoms_96x363_363x3072();

int setup_suite_sgemm_RNN() {
    srand(0xDEADBEEF);
    return 0;
}

int teardown_suite_sgemm_RNN() {
    return 0;
}

void suite_sgemm_RNN() {
    CU_pSuite suite = CU_add_suite("sgemm RNN", setup_suite_sgemm_RNN, teardown_suite_sgemm_RNN);

    CU_add_test(suite, "ones 16x2 * 2x64" , test_sgemm_RNN_ones_16x2_2x64 );
    CU_add_test(suite, "ones 16x2 * 2x128", test_sgemm_RNN_ones_16x2_2x128);
    CU_add_test(suite, "ones 16x2 * 2x192", test_sgemm_RNN_ones_16x2_2x192);
    CU_add_test(suite, "ones 16x2 * 2x256", test_sgemm_RNN_ones_16x2_2x256);
    CU_add_test(suite, "ones 16x2 * 2x320", test_sgemm_RNN_ones_16x2_2x320);
    CU_add_test(suite, "ones 16x2 * 2x384", test_sgemm_RNN_ones_16x2_2x384);
    CU_add_test(suite, "ones 16x2 * 2x448", test_sgemm_RNN_ones_16x2_2x448);
    CU_add_test(suite, "ones 32x2 * 2x64" , test_sgemm_RNN_ones_32x2_2x64 );
    CU_add_test(suite, "ones 32x2 * 2x128", test_sgemm_RNN_ones_32x2_2x128);
    CU_add_test(suite, "ones 32x2 * 2x192", test_sgemm_RNN_ones_32x2_2x192);
    CU_add_test(suite, "ones 32x2 * 2x256", test_sgemm_RNN_ones_32x2_2x256);
    CU_add_test(suite, "ones 32x2 * 2x320", test_sgemm_RNN_ones_32x2_2x320);
    CU_add_test(suite, "ones 32x2 * 2x384", test_sgemm_RNN_ones_32x2_2x384);
    CU_add_test(suite, "ones 32x2 * 2x448", test_sgemm_RNN_ones_32x2_2x448);
    CU_add_test(suite, "ones 48x2 * 2x64" , test_sgemm_RNN_ones_48x2_2x64 );
    CU_add_test(suite, "ones 48x2 * 2x128", test_sgemm_RNN_ones_48x2_2x128);
    CU_add_test(suite, "ones 48x2 * 2x192", test_sgemm_RNN_ones_48x2_2x192);
    CU_add_test(suite, "ones 48x2 * 2x256", test_sgemm_RNN_ones_48x2_2x256);
    CU_add_test(suite, "ones 48x2 * 2x320", test_sgemm_RNN_ones_48x2_2x320);
    CU_add_test(suite, "ones 48x2 * 2x384", test_sgemm_RNN_ones_48x2_2x384);
    CU_add_test(suite, "ones 48x2 * 2x448", test_sgemm_RNN_ones_48x2_2x448);
    CU_add_test(suite, "ones 96x363 * 363x3072", test_sgemm_RNN_ones_96x363_363x3072);

    CU_add_test(suite, "randoms 16x2 * 2x64" , test_sgemm_RNN_randoms_16x2_2x64 );
    CU_add_test(suite, "randoms 16x2 * 2x128", test_sgemm_RNN_randoms_16x2_2x128);
    CU_add_test(suite, "randoms 16x2 * 2x192", test_sgemm_RNN_randoms_16x2_2x192);
    CU_add_test(suite, "randoms 16x2 * 2x256", test_sgemm_RNN_randoms_16x2_2x256);
    CU_add_test(suite, "randoms 16x2 * 2x320", test_sgemm_RNN_randoms_16x2_2x320);
    CU_add_test(suite, "randoms 16x2 * 2x384", test_sgemm_RNN_randoms_16x2_2x384);
    CU_add_test(suite, "randoms 16x2 * 2x448", test_sgemm_RNN_randoms_16x2_2x448);
    CU_add_test(suite, "randoms 32x2 * 2x64" , test_sgemm_RNN_randoms_32x2_2x64 );
    CU_add_test(suite, "randoms 32x2 * 2x128", test_sgemm_RNN_randoms_32x2_2x128);
    CU_add_test(suite, "randoms 32x2 * 2x192", test_sgemm_RNN_randoms_32x2_2x192);
    CU_add_test(suite, "randoms 32x2 * 2x256", test_sgemm_RNN_randoms_32x2_2x256);
    CU_add_test(suite, "randoms 32x2 * 2x320", test_sgemm_RNN_randoms_32x2_2x320);
    CU_add_test(suite, "randoms 32x2 * 2x384", test_sgemm_RNN_randoms_32x2_2x384);
    CU_add_test(suite, "randoms 32x2 * 2x448", test_sgemm_RNN_randoms_32x2_2x448);
    CU_add_test(suite, "randoms 48x2 * 2x64" , test_sgemm_RNN_randoms_48x2_2x64 );
    CU_add_test(suite, "randoms 48x2 * 2x128", test_sgemm_RNN_randoms_48x2_2x128);
    CU_add_test(suite, "randoms 48x2 * 2x192", test_sgemm_RNN_randoms_48x2_2x192);
    CU_add_test(suite, "randoms 48x2 * 2x256", test_sgemm_RNN_randoms_48x2_2x256);
    CU_add_test(suite, "randoms 48x2 * 2x320", test_sgemm_RNN_randoms_48x2_2x320);
    CU_add_test(suite, "randoms 48x2 * 2x384", test_sgemm_RNN_randoms_48x2_2x384);
    CU_add_test(suite, "randoms 48x2 * 2x448", test_sgemm_RNN_randoms_48x2_2x448);
    CU_add_test(suite, "randoms 96x363 * 363x3072", test_sgemm_RNN_randoms_96x363_363x3072);
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
    {
        int i, j;
        for (i = 0; i < M; ++i)
            for (j = 0; j < N; ++j)
                CU_ASSERT_EQUAL(K+1, (int)C[i*N+j]);
    }
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

void test_sgemm_RNN_ones_16x2_2x64()  { test_sgemm_RNN_ones(16,  64, 2); }
void test_sgemm_RNN_ones_16x2_2x128() { test_sgemm_RNN_ones(16, 128, 2); }
void test_sgemm_RNN_ones_16x2_2x192() { test_sgemm_RNN_ones(16, 192, 2); }
void test_sgemm_RNN_ones_16x2_2x256() { test_sgemm_RNN_ones(16, 256, 2); }
void test_sgemm_RNN_ones_16x2_2x320() { test_sgemm_RNN_ones(16, 320, 2); }
void test_sgemm_RNN_ones_16x2_2x384() { test_sgemm_RNN_ones(16, 384, 2); }
void test_sgemm_RNN_ones_16x2_2x448() { test_sgemm_RNN_ones(16, 448, 2); }
void test_sgemm_RNN_ones_32x2_2x64 () { test_sgemm_RNN_ones(32,  64, 2); }
void test_sgemm_RNN_ones_32x2_2x128() { test_sgemm_RNN_ones(32, 128, 2); }
void test_sgemm_RNN_ones_32x2_2x192() { test_sgemm_RNN_ones(32, 192, 2); }
void test_sgemm_RNN_ones_32x2_2x256() { test_sgemm_RNN_ones(32, 256, 2); }
void test_sgemm_RNN_ones_32x2_2x320() { test_sgemm_RNN_ones(32, 320, 2); }
void test_sgemm_RNN_ones_32x2_2x384() { test_sgemm_RNN_ones(32, 384, 2); }
void test_sgemm_RNN_ones_32x2_2x448() { test_sgemm_RNN_ones(32, 448, 2); }
void test_sgemm_RNN_ones_48x2_2x64 () { test_sgemm_RNN_ones(48,  64, 2); }
void test_sgemm_RNN_ones_48x2_2x128() { test_sgemm_RNN_ones(48, 128, 2); }
void test_sgemm_RNN_ones_48x2_2x192() { test_sgemm_RNN_ones(48, 192, 2); }
void test_sgemm_RNN_ones_48x2_2x256() { test_sgemm_RNN_ones(48, 256, 2); }
void test_sgemm_RNN_ones_48x2_2x320() { test_sgemm_RNN_ones(48, 320, 2); }
void test_sgemm_RNN_ones_48x2_2x384() { test_sgemm_RNN_ones(48, 384, 2); }
void test_sgemm_RNN_ones_48x2_2x448() { test_sgemm_RNN_ones(48, 448, 2); }
void test_sgemm_RNN_ones_96x363_363x3072() { test_sgemm_RNN_ones(96, 3072, 363); }

static float rand_in_range(float from, float to) {
    return ((float)rand() / RAND_MAX) * (to - from) + from;
}

static float* mkl_malloc_randoms(const int m, const int n) {
    float* p = mkl_malloc(m * n * sizeof(float), 4096);
    int i = 0;
#pragma omp parallel for private(i)
    for (i = 0; i < m * n; ++i) p[i] = rand_in_range(-1.0, 1.0);
    return p;
}

static float* mkl_malloc_with_copy(const int m, const int n, const float* src) {
    float* p = mkl_malloc(m * n * sizeof(float), 4096);
    memcpy(p, src, m * n * sizeof(float));
    return p;
}

static void test_sgemm_RNN_randoms(const int M, const int N, const int K) {
    float* A = mkl_malloc_randoms(M, K);
    float* B = mkl_malloc_randoms(K, N);
    float* C = mkl_malloc_randoms(M, N);
    float* R = mkl_malloc_with_copy(M, N, C);
    const float alpha = rand_in_range(-1.0, 1.0);
    const float beta = rand_in_range(-1.0, 1.0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
    {
        int i, j, k;
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float acc = 0;
                for (k = 0; k < K; ++k) acc += A[i*K+k] * B[k*N+j];
                R[i*N+j] = alpha * acc + beta * R[i*N+j];
            }
        }
    }
    {
        float maximum_abs_error = 0;
        int i, j;
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                if (maximum_abs_error < abs(R[i*N+j] - C[i*N+j]))
                    maximum_abs_error = abs(R[i*N+j] - C[i*N+j]);
            }
        }
        CU_ASSERT_DOUBLE_EQUAL(maximum_abs_error, 0, 0.001);
    }
    mkl_free(R);
    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
}

void test_sgemm_RNN_randoms_16x2_2x64()  { test_sgemm_RNN_randoms(16,  64, 2); }
void test_sgemm_RNN_randoms_16x2_2x128() { test_sgemm_RNN_randoms(16, 128, 2); }
void test_sgemm_RNN_randoms_16x2_2x192() { test_sgemm_RNN_randoms(16, 192, 2); }
void test_sgemm_RNN_randoms_16x2_2x256() { test_sgemm_RNN_randoms(16, 256, 2); }
void test_sgemm_RNN_randoms_16x2_2x320() { test_sgemm_RNN_randoms(16, 320, 2); }
void test_sgemm_RNN_randoms_16x2_2x384() { test_sgemm_RNN_randoms(16, 384, 2); }
void test_sgemm_RNN_randoms_16x2_2x448() { test_sgemm_RNN_randoms(16, 448, 2); }
void test_sgemm_RNN_randoms_32x2_2x64 () { test_sgemm_RNN_randoms(32,  64, 2); }
void test_sgemm_RNN_randoms_32x2_2x128() { test_sgemm_RNN_randoms(32, 128, 2); }
void test_sgemm_RNN_randoms_32x2_2x192() { test_sgemm_RNN_randoms(32, 192, 2); }
void test_sgemm_RNN_randoms_32x2_2x256() { test_sgemm_RNN_randoms(32, 256, 2); }
void test_sgemm_RNN_randoms_32x2_2x320() { test_sgemm_RNN_randoms(32, 320, 2); }
void test_sgemm_RNN_randoms_32x2_2x384() { test_sgemm_RNN_randoms(32, 384, 2); }
void test_sgemm_RNN_randoms_32x2_2x448() { test_sgemm_RNN_randoms(32, 448, 2); }
void test_sgemm_RNN_randoms_48x2_2x64 () { test_sgemm_RNN_randoms(48,  64, 2); }
void test_sgemm_RNN_randoms_48x2_2x128() { test_sgemm_RNN_randoms(48, 128, 2); }
void test_sgemm_RNN_randoms_48x2_2x192() { test_sgemm_RNN_randoms(48, 192, 2); }
void test_sgemm_RNN_randoms_48x2_2x256() { test_sgemm_RNN_randoms(48, 256, 2); }
void test_sgemm_RNN_randoms_48x2_2x320() { test_sgemm_RNN_randoms(48, 320, 2); }
void test_sgemm_RNN_randoms_48x2_2x384() { test_sgemm_RNN_randoms(48, 384, 2); }
void test_sgemm_RNN_randoms_48x2_2x448() { test_sgemm_RNN_randoms(48, 448, 2); }
void test_sgemm_RNN_randoms_96x363_363x3072() { test_sgemm_RNN_randoms(96, 3072, 363); }
