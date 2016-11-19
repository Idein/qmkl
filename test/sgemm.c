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
#include <math.h>
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

static float mf_minimum_absolute_error(float *C1, float *C2, const int P, const int R)
{
	int i, j;
	float minimum_error = +INFINITY;
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j ++) {
			float error = fabs(C1[i * R + j] - C2[i * R + j]);
			if (error < minimum_error)
				minimum_error = error;
		}
	}
	return minimum_error;
}

static float mf_maximum_absolute_error(float *C1, float *C2, const int P, const int R)
{
	int i, j;
	float maximum_error = 0.0;
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j ++) {
			float error = fabs(C1[i * R + j] - C2[i * R + j]);
			if (error > maximum_error)
				maximum_error = error;
		}
	}
	return maximum_error;
}

static float mf_minimum_relative_error(float *C1, float *C2, const int P, const int R)
{
	int i, j;
	float minimum_error = +INFINITY;
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j ++) {
			float error = fabs((C1[i * R + j] - C2[i * R + j]) / C2[i * R + j]);
			if (error < minimum_error)
				minimum_error = error;
		}
	}
	return minimum_error;
}

static float mf_maximum_relative_error(float *C1, float *C2, const int P, const int R)
{
	int i, j;
	float maximum_error = 0.0;
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j ++) {
			float error = fabs((C1[i * R + j] - C2[i * R + j]) / C2[i * R + j]);
			if (error > maximum_error)
				maximum_error = error;
		}
	}
	return maximum_error;
}

static void mf_sgemm(float *A, float *B, float *C, const int P, const int Q, const int R, const float ALPHA, const float BETA)
{
	int i, j, k;

#pragma omp parallel for private(i, j, k)
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j ++) {
			float sum = 0.0;
			for (k = 0; k < Q; k ++) {
				sum += A[i * Q + k] * B[k * R + j];
			}
			C[i * R + j] = ALPHA * sum + BETA * C[i * R + j];
		}
	}
}

int main()
{
	const unsigned NREG = 16;
	const unsigned P = 96;
	const unsigned Q = 363;
	const unsigned R = 3072;
	float *A, *B, *C, *C_ref;
	float ALPHA;
	float BETA;
	struct timeval start, end;

	if (P % 16 != 0) {
		fprintf(stderr, "error: P must be a multiple of 16\n");
		exit(EXIT_FAILURE);
	}

	if (R % NREG != 0) {
		fprintf(stderr, "error: R must be a multiple of NREG\n");
		exit(EXIT_FAILURE);
	}

	A     = mkl_malloc(P * Q * (32 / 8), 4096);
	B     = mkl_malloc(Q * R * (32 / 8), 4096);
	C     = mkl_malloc(P * R * (32 / 8), 4096);
	C_ref = mkl_malloc(P * R * (32 / 8), 4096);
	mf_srandom();
	mf_init_random(A, P, Q);
	mf_init_random(B, Q, R);
	mf_init_random(C, P, R);
	memcpy(C_ref, C, P * R * (32 / 8));
	mf_init_random(&ALPHA, 1, 1);
	mf_init_random(&BETA,  1, 1);

	printf("P = %d\n", P);
	printf("Q = %d\n", Q);
	printf("R = %d\n", R);
	printf("ALPHA = %f\n", ALPHA);
	printf("BETA = %f\n", BETA);
	printf("==== sgemm example (ALPHA * %dx%d * %dx%d + BETA * %dx%d) ====\n", P, Q, Q, R, P, R);

	printf("GPU: "); fflush(stdout);
	gettimeofday(&start, NULL);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, R, Q, ALPHA, A, Q, B, R, BETA, C, R);
	gettimeofday(&end, NULL);
	printf("%g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, (2 * P * Q * R + 3 * P * R) / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

	printf("CPU (%d threads): ", omp_get_max_threads()); fflush(stdout);
	gettimeofday(&start, NULL);
	mf_sgemm(A, B, C_ref, P, Q, R, ALPHA, BETA);
	gettimeofday(&end, NULL);
	printf("%g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, (2 * P * Q * R + 3 * P * R) / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

	printf("Minimum absolute error: %g\n", mf_minimum_absolute_error(C_ref, C, P, R));
	printf("Maximum absolute error: %g\n", mf_maximum_absolute_error(C_ref, C, P, R));
	printf("Minimum relative error: %g\n", mf_minimum_relative_error(C_ref, C, P, R));
	printf("Maximum relative error: %g\n", mf_maximum_relative_error(C_ref, C, P, R));

	mkl_free(C_ref);
	mkl_free(C);
	mkl_free(B);
	mkl_free(A);
	return 0;
}
