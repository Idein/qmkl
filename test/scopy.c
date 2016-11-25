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
		size_t offset, len;

		offset = ((n / num_threads) * thread_num) / sizeof(*x);
		len = n / num_threads;
		if (thread_num == num_threads - 1)
			len = n - (n / num_threads) * (num_threads - 1);

		memcpy(y + offset, x + offset, len);
	}
}

int main()
{
	const int n = 4096 * 1024;
	float *x, *y, *y_ref;
	struct timeval start, end;

	x     = mkl_malloc(n * sizeof(*x),     4096);
	y     = mkl_malloc(n * sizeof(*y),     4096);
	y_ref = mkl_malloc(n * sizeof(*y_ref), 4096);

	mf_srandom();
	mf_init_random(x, 1, n);

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

	mkl_free(y_ref);
	mkl_free(y);
	mkl_free(x);
	return 0;
}
