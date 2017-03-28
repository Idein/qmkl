#include "config.h"
#include <unistd.h>
#include <stdlib.h>
#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include <sys/time.h>
#include "mkl.h"

static double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + t.tv_usec * 1e-6;
}

static void* host_malloc(size_t size) { return malloc(size); }
static void  host_free(void* p) { if (p) free(p); }
static void* gpu_malloc(size_t size) { return mkl_malloc(size, 4096); }
static void  gpu_free(void* p) { if (p) mkl_free(p); }

/* invalidate_cpu_l2c - may invalidate CPU-unified L2C */
static void invalidate_cpu_l2c()
{
    int i;
    const int NLOOPS = 10;
    size_t j;
    const size_t SIZE = 16 * 1024 * 1024;
    int *p = host_malloc(SIZE);
    const size_t LEN = SIZE / sizeof(*p);
    for (i = 0; i < NLOOPS; i ++) {
        p[LEN - 1] = p[0];
        for (j = 0; j < LEN - 1; j ++)
            p[j] = p[j + 1];
    }
    host_free(p);
}

static float rand_float_in_range(float from, float to) {
    return ((float)rand() / RAND_MAX) * (to - from) + from;
}

const size_t N = 1024 * 1024;
static float* gpu_src_memory = NULL;
static float* host_src_memory = NULL;
static float expected = 0;

static int setup_suite() {
    srand(0xDEADBEEF);
    gpu_src_memory = (float*)gpu_malloc(N * sizeof(float));
    size_t i = 0;
#pragma omp parallel for private(i)
    for (i = 0; i < N; ++i) gpu_src_memory[i] = rand_float_in_range(-1.0, 1.0);
    host_src_memory = (float*)host_malloc(N*sizeof(float));
    memcpy(host_src_memory, gpu_src_memory, N*sizeof(float));
    for (i = 0; i < N; ++i)
        expected += host_src_memory[i];
    return 0;
}

static int teardown_suite() {
    gpu_free(gpu_src_memory);
    host_free(host_src_memory);
    expected = 0;
    return 0;
}

#define TEST_MEMCPY(FROM, TO)                                           \
    static void test_memcpy_##FROM##_to_##TO() {                        \
        float* TO##_dst_memory = (float*)TO##_malloc(N*sizeof(float));  \
        invalidate_cpu_l2c();                                           \
        double start = get_time();                                      \
        memcpy(TO##_dst_memory, FROM##_src_memory, N*sizeof(float));    \
        double end = get_time();                                        \
        float actual = 0;                                               \
        size_t i = 0;                                                   \
        for (i = 0; i < N; ++i) {                                       \
            actual += TO##_dst_memory[i];                               \
        }                                                               \
        printf("\nmemcpy(4MB): %lf sec (%lf)\n",                        \
               end - start,                                             \
               N * sizeof(float) / (end - start) / 1024 / 1024          \
        );                                                              \
        CU_ASSERT_DOUBLE_EQUAL(actual, expected, 0.001);                \
        TO##_free(TO##_dst_memory);                                     \
    }

TEST_MEMCPY(host, host);
TEST_MEMCPY(host, gpu);
TEST_MEMCPY(gpu, host);
TEST_MEMCPY(gpu, gpu);

static void suite_memcpy() {
    CU_pSuite suite = CU_add_suite(
        "memcpy",
        setup_suite,
        teardown_suite
    );

    CU_add_test(suite, "Host -> Host", test_memcpy_host_to_host);
    CU_add_test(suite, "Host -> GPU", test_memcpy_host_to_gpu);
    CU_add_test(suite, "GPU -> Host", test_memcpy_gpu_to_host);
    CU_add_test(suite, "GPU -> GPU", test_memcpy_gpu_to_gpu);
}

#define TEST_MEMSET(TO)                                                 \
    static void test_memset_to_##TO() {                                 \
        float* TO##_dst_memory = (float*)TO##_malloc(N*sizeof(float));  \
        invalidate_cpu_l2c();                                           \
        double start = get_time();                                      \
        memset(TO##_dst_memory, 0, N*sizeof(float));                    \
        double end = get_time();                                        \
        float actual = 0;                                               \
        size_t i = 0;                                                   \
        for (i = 0; i < N; ++i) {                                       \
            actual += TO##_dst_memory[i];                               \
        }                                                               \
        printf("\nmemset(4MB): %lf sec (%lf)\n",                        \
               end - start,                                             \
               N * sizeof(float) / (end - start) / 1024 / 1024          \
        );                                                              \
        CU_ASSERT_DOUBLE_EQUAL(actual, 0, 0.001);                       \
        TO##_free(TO##_dst_memory);                                     \
    }

TEST_MEMSET(host);
TEST_MEMSET(gpu);

static void suite_memset() {
    CU_pSuite suite = CU_add_suite(
        "memset",
        setup_suite,
        teardown_suite
    );

    CU_add_test(suite, "0 -> Host", test_memset_to_host);
    CU_add_test(suite, "0 -> GPU", test_memset_to_gpu);
}

#define TEST_SUMALL(FROM)                                               \
    static void test_sumall_##FROM() {                                  \
        invalidate_cpu_l2c();                                           \
        double start = get_time();                                      \
        float actual = 0;                                               \
        size_t i = 0;                                                   \
        for (i = 0; i < N; ++i) {                                       \
            actual += FROM##_src_memory[i];                             \
        }                                                               \
        double end = get_time();                                        \
        printf("\nsumall(4MB): %lf sec (%lf)\n",                        \
               end - start,                                             \
               N * sizeof(float) / (end - start) / 1024 / 1024          \
        );                                                              \
        CU_ASSERT_DOUBLE_EQUAL(actual, expected, 0.001);                \
    }

TEST_SUMALL(host);
TEST_SUMALL(gpu);

static void suite_sumall() {
    CU_pSuite suite = CU_add_suite(
        "sumall",
        setup_suite,
        teardown_suite
    );

    CU_add_test(suite, "Host -> x", test_sumall_host);
    CU_add_test(suite, "GPU -> x", test_sumall_gpu);
}

int main() {
    CU_initialize_registry();

    suite_memcpy();
    suite_memset();
    suite_sumall();

    isatty(fileno(stdout)) ? CU_console_run_tests() : CU_basic_run_tests();
    const unsigned int result = CU_get_number_of_failures();
    CU_cleanup_registry();
    return (result ? 1 : 0);
}
