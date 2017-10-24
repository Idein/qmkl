/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include "mkl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#ifdef _HAVE_NEON_
#include <arm_neon.h>
#endif /* _HAVE_NEON_ */

static void mf_srandom()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    srandom(tv.tv_sec ^ tv.tv_usec);
}

static void mf_init_random(float *p, const int height, const int width)
{
    int i, j;

    for (i = 0; i < height; i ++) {
        for (j = 0; j < width; j ++) {
            const float f = (random() % 100000) / 1357.9;
            p[i * width + j] = f / 2.0 - f;
        }
    }
}

static void mf_vsAbs(const MKL_INT n, const float *a, float *y)
{
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < n; i ++)
        y[i] = fabsf(a[i]);
}

#ifdef _HAVE_NEON_
static void mf_vsAbs_neon(const MKL_INT n, const float *a, float *y)
{
    int i;

#pragma omp parallel for private(i)
    for (i = 0; i < n; i += 4 * 16) {
        float32x4_t va[16], vy[16];
        va[ 0] = vld1q_f32(a + i + 4 *  0);
        va[ 1] = vld1q_f32(a + i + 4 *  1);
        va[ 2] = vld1q_f32(a + i + 4 *  2);
        va[ 3] = vld1q_f32(a + i + 4 *  3);
        va[ 4] = vld1q_f32(a + i + 4 *  4);
        va[ 5] = vld1q_f32(a + i + 4 *  5);
        va[ 6] = vld1q_f32(a + i + 4 *  6);
        va[ 7] = vld1q_f32(a + i + 4 *  7);
        va[ 8] = vld1q_f32(a + i + 4 *  8);
        va[ 9] = vld1q_f32(a + i + 4 *  9);
        va[10] = vld1q_f32(a + i + 4 * 10);
        va[11] = vld1q_f32(a + i + 4 * 11);
        va[12] = vld1q_f32(a + i + 4 * 12);
        va[13] = vld1q_f32(a + i + 4 * 13);
        va[14] = vld1q_f32(a + i + 4 * 14);
        va[15] = vld1q_f32(a + i + 4 * 15);
        vy[ 0] = vabsq_f32(va[ 0]);
        vy[ 1] = vabsq_f32(va[ 1]);
        vy[ 2] = vabsq_f32(va[ 2]);
        vy[ 3] = vabsq_f32(va[ 3]);
        vy[ 4] = vabsq_f32(va[ 4]);
        vy[ 5] = vabsq_f32(va[ 5]);
        vy[ 6] = vabsq_f32(va[ 6]);
        vy[ 7] = vabsq_f32(va[ 7]);
        vy[ 8] = vabsq_f32(va[ 8]);
        vy[ 9] = vabsq_f32(va[ 9]);
        vy[10] = vabsq_f32(va[10]);
        vy[11] = vabsq_f32(va[11]);
        vy[12] = vabsq_f32(va[12]);
        vy[13] = vabsq_f32(va[13]);
        vy[14] = vabsq_f32(va[14]);
        vy[15] = vabsq_f32(va[15]);
        vst1q_f32(y + i + 4 *  0, vy[ 0]);
        vst1q_f32(y + i + 4 *  1, vy[ 1]);
        vst1q_f32(y + i + 4 *  2, vy[ 2]);
        vst1q_f32(y + i + 4 *  3, vy[ 3]);
        vst1q_f32(y + i + 4 *  4, vy[ 4]);
        vst1q_f32(y + i + 4 *  5, vy[ 5]);
        vst1q_f32(y + i + 4 *  6, vy[ 6]);
        vst1q_f32(y + i + 4 *  7, vy[ 7]);
        vst1q_f32(y + i + 4 *  8, vy[ 8]);
        vst1q_f32(y + i + 4 *  9, vy[ 9]);
        vst1q_f32(y + i + 4 * 10, vy[10]);
        vst1q_f32(y + i + 4 * 11, vy[11]);
        vst1q_f32(y + i + 4 * 12, vy[12]);
        vst1q_f32(y + i + 4 * 13, vy[13]);
        vst1q_f32(y + i + 4 * 14, vy[14]);
        vst1q_f32(y + i + 4 * 15, vy[15]);
    }
}
#endif /* _HAVE_NEON_ */

int main()
{
    const int n = 4096 * 512 * 3;
    float *a, *y, *y_ref;
#ifdef _HAVE_NEON_
    float *y_neon;
#endif /* _HAVE_NEON_ */
    struct timeval start, end;

    a     = mkl_malloc(n * sizeof(*a),     4096);
    y     = mkl_malloc(n * sizeof(*y),     4096);
    y_ref = mkl_malloc(n * sizeof(*y_ref), 4096);
#ifdef _HAVE_NEON_
    y_neon = mkl_malloc(n * sizeof(*y_neon), 4096);
#endif /* _HAVE_NEON_ */

    mf_srandom();
    mf_init_random(a, 1, n);

    printf("n = %d\n", n);
    printf("==== vsAbs example (y = abs(a)) ====\n");

    printf("GPU: "); fflush(stdout);
    gettimeofday(&start, NULL);
    vsAbs(n, a, y);
    gettimeofday(&end, NULL);
    printf("%g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, n / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

    printf("CPU (%d threads): ", omp_get_max_threads()); fflush(stdout);
    gettimeofday(&start, NULL);
    mf_vsAbs(n, a, y_ref);
    gettimeofday(&end, NULL);
    printf("%g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, n / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

    if (memcmp(y, y_ref, n * sizeof(*y))) {
        int i;
        for (i = 0; i < n; i ++) {
            if (y[i] != y_ref[i]) {
                printf("GPU and CPU differ at i=%d (%f vs. %f)\n", i, y[i], y_ref[i]);
                break;
            }
        }
    }

#ifdef _HAVE_NEON_
    printf("CPU with NEON (%d threads): ", omp_get_max_threads()); fflush(stdout);
    gettimeofday(&start, NULL);
    mf_vsAbs_neon(n, a, y_neon);
    gettimeofday(&end, NULL);
    printf("%g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, n / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

    if (memcmp(y_neon, y_ref, n * sizeof(*y))) {
        int i;
        for (i = 0; i < n; i ++) {
            if (y_neon[i] != y_ref[i]) {
                printf("GPU and CPU differ at i=%d (%f vs. %f)\n", i, y_neon[i], y_ref[i]);
                break;
            }
        }
    }

    mkl_free(y_neon);
#endif /* _HAVE_NEON_ */
    mkl_free(y_ref);
    mkl_free(y);
    mkl_free(a);
    return 0;
}
