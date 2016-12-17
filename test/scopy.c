/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
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

	for (i = 0; i < height; i ++)
		for (j = 0; j < width; j ++)
			p[i * width + j] = (random() % 100000) / 1357.9;
}

static void mf_scopy(const MKL_INT n, const float *x, float *y)
{
#pragma omp parallel
	{
		int thread_num = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		size_t offset, size;

		offset = (n / num_threads) * thread_num * sizeof(*x);
		size = n / num_threads * sizeof(*x);

		memcpy(((uint8_t*) y) + offset, ((uint8_t*) x) + offset, size);
	}
}

#ifdef _HAVE_NEON_
static void mf_scopy_neon(const MKL_INT n, const float *x, float *y)
{
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < n; i += 4 * 16) {
		float32x4_t vx[16];
		vx[ 0] = vld1q_f32(x + i + 4 *  0);
		vx[ 1] = vld1q_f32(x + i + 4 *  1);
		vx[ 2] = vld1q_f32(x + i + 4 *  2);
		vx[ 3] = vld1q_f32(x + i + 4 *  3);
		vx[ 4] = vld1q_f32(x + i + 4 *  4);
		vx[ 5] = vld1q_f32(x + i + 4 *  5);
		vx[ 6] = vld1q_f32(x + i + 4 *  6);
		vx[ 7] = vld1q_f32(x + i + 4 *  7);
		vx[ 8] = vld1q_f32(x + i + 4 *  8);
		vx[ 9] = vld1q_f32(x + i + 4 *  9);
		vx[10] = vld1q_f32(x + i + 4 * 10);
		vx[11] = vld1q_f32(x + i + 4 * 11);
		vx[12] = vld1q_f32(x + i + 4 * 12);
		vx[13] = vld1q_f32(x + i + 4 * 13);
		vx[14] = vld1q_f32(x + i + 4 * 14);
		vx[15] = vld1q_f32(x + i + 4 * 15);
		vst1q_f32(y + i + 4 *  0, vx[ 0]);
		vst1q_f32(y + i + 4 *  1, vx[ 1]);
		vst1q_f32(y + i + 4 *  2, vx[ 2]);
		vst1q_f32(y + i + 4 *  3, vx[ 3]);
		vst1q_f32(y + i + 4 *  4, vx[ 4]);
		vst1q_f32(y + i + 4 *  5, vx[ 5]);
		vst1q_f32(y + i + 4 *  6, vx[ 6]);
		vst1q_f32(y + i + 4 *  7, vx[ 7]);
		vst1q_f32(y + i + 4 *  8, vx[ 8]);
		vst1q_f32(y + i + 4 *  9, vx[ 9]);
		vst1q_f32(y + i + 4 * 10, vx[10]);
		vst1q_f32(y + i + 4 * 11, vx[11]);
		vst1q_f32(y + i + 4 * 12, vx[12]);
		vst1q_f32(y + i + 4 * 13, vx[13]);
		vst1q_f32(y + i + 4 * 14, vx[14]);
		vst1q_f32(y + i + 4 * 15, vx[15]);
	}
}
#endif /* _HAVE_NEON_ */

int main()
{
	const int n = 4096 * 1024;
	float *x, *y, *y_ref;
#ifdef _HAVE_NEON_
	float *y_neon;
#endif /* _HAVE_NEON_ */
	struct timeval start, end;

	x     = mkl_malloc(n * sizeof(*x),     4096);
	y     = mkl_malloc(n * sizeof(*y),     4096);
	y_ref = mkl_malloc(n * sizeof(*y_ref), 4096);
#ifdef _HAVE_NEON_
	y_neon = mkl_malloc(n * sizeof(*y_neon), 4096);
#endif /* _HAVE_NEON_ */

	mf_srandom();
	mf_init_random(x, 1, n);
	/* To detect insufficient copies... */
	mf_init_random(y, 1, n);
	mf_init_random(y_ref, 1, n);
#ifdef _HAVE_NEON_
	mf_init_random(y_neon, 1, n);
#endif /* _HAVE_NEON_ */

	printf("n = %d\n", n);
	printf("==== scopy example (y = x) ====\n");

	printf("GPU: "); fflush(stdout);
	gettimeofday(&start, NULL);
	cblas_scopy(n, x, 1, y, 1);
	gettimeofday(&end, NULL);
	printf("%g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, n / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

	printf("CPU (%d threads): ", omp_get_max_threads()); fflush(stdout);
	gettimeofday(&start, NULL);
	mf_scopy(n, x, y_ref);
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
	mf_scopy_neon(n, x, y_neon);
	gettimeofday(&end, NULL);
	printf("%g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, n / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

	if (memcmp(y_neon, y_ref, n * sizeof(*y))) {
		int i;
		for (i = 0; i < n; i ++) {
			if (y_neon[i] != y_ref[i]) {
				printf("NEON and CPU differ at i=%d (%f vs. %f)\n", i, y_neon[i], y_ref[i]);
				break;
			}
		}
	}

	mkl_free(y_neon);
#endif /* _HAVE_NEON_ */
	mkl_free(y_ref);
	mkl_free(y);
	mkl_free(x);
	return 0;
}
