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

#ifdef _HAVE_NEON_
#include <arm_neon.h>
#endif /* _HAVE_NEON_ */

static void mf_init_constant(float *p, const int height, const int width, const float c)
{
	int i, j;

	for (i = 0; i < height; i ++)
		for (j = 0; j < width; j ++)
			p[i * width + j] = c;
}

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

#ifdef _HAVE_NEON_
static void mf_sgemm_neon(float *A, float *B, float *C, const int P, const int Q, const int R, const float ALPHA, const float BETA)
{
	int i, j, k;

#pragma omp parallel for private(i, j, k)
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j += 4 * 16) {
			float32x4_t vsum[16], vc[16];
			vsum[ 0] = vdupq_n_f32(0.0f);
			vsum[ 1] = vdupq_n_f32(0.0f);
			vsum[ 2] = vdupq_n_f32(0.0f);
			vsum[ 3] = vdupq_n_f32(0.0f);
			vsum[ 4] = vdupq_n_f32(0.0f);
			vsum[ 5] = vdupq_n_f32(0.0f);
			vsum[ 6] = vdupq_n_f32(0.0f);
			vsum[ 7] = vdupq_n_f32(0.0f);
			vsum[ 8] = vdupq_n_f32(0.0f);
			vsum[ 9] = vdupq_n_f32(0.0f);
			vsum[10] = vdupq_n_f32(0.0f);
			vsum[11] = vdupq_n_f32(0.0f);
			vsum[12] = vdupq_n_f32(0.0f);
			vsum[13] = vdupq_n_f32(0.0f);
			vsum[14] = vdupq_n_f32(0.0f);
			vsum[15] = vdupq_n_f32(0.0f);
			for (k = 0; k < Q; k ++) {
				float a = A[i * Q + k];
				float32x4_t vb[16];
				vb[ 0] = vld1q_f32(B + k * R + j + 4 *  0);
				vb[ 1] = vld1q_f32(B + k * R + j + 4 *  1);
				vb[ 2] = vld1q_f32(B + k * R + j + 4 *  2);
				vb[ 3] = vld1q_f32(B + k * R + j + 4 *  3);
				vb[ 4] = vld1q_f32(B + k * R + j + 4 *  4);
				vb[ 5] = vld1q_f32(B + k * R + j + 4 *  5);
				vb[ 6] = vld1q_f32(B + k * R + j + 4 *  6);
				vb[ 7] = vld1q_f32(B + k * R + j + 4 *  7);
				vb[ 8] = vld1q_f32(B + k * R + j + 4 *  8);
				vb[ 9] = vld1q_f32(B + k * R + j + 4 *  9);
				vb[10] = vld1q_f32(B + k * R + j + 4 * 10);
				vb[11] = vld1q_f32(B + k * R + j + 4 * 11);
				vb[12] = vld1q_f32(B + k * R + j + 4 * 12);
				vb[13] = vld1q_f32(B + k * R + j + 4 * 13);
				vb[14] = vld1q_f32(B + k * R + j + 4 * 14);
				vb[15] = vld1q_f32(B + k * R + j + 4 * 15);
				vsum[ 0] = vmlaq_n_f32(vsum[ 0], vb[ 0], a);
				vsum[ 1] = vmlaq_n_f32(vsum[ 1], vb[ 1], a);
				vsum[ 2] = vmlaq_n_f32(vsum[ 2], vb[ 2], a);
				vsum[ 3] = vmlaq_n_f32(vsum[ 3], vb[ 3], a);
				vsum[ 4] = vmlaq_n_f32(vsum[ 4], vb[ 4], a);
				vsum[ 5] = vmlaq_n_f32(vsum[ 5], vb[ 5], a);
				vsum[ 6] = vmlaq_n_f32(vsum[ 6], vb[ 6], a);
				vsum[ 7] = vmlaq_n_f32(vsum[ 7], vb[ 7], a);
				vsum[ 8] = vmlaq_n_f32(vsum[ 8], vb[ 8], a);
				vsum[ 9] = vmlaq_n_f32(vsum[ 9], vb[ 9], a);
				vsum[10] = vmlaq_n_f32(vsum[10], vb[10], a);
				vsum[11] = vmlaq_n_f32(vsum[11], vb[11], a);
				vsum[12] = vmlaq_n_f32(vsum[12], vb[12], a);
				vsum[13] = vmlaq_n_f32(vsum[13], vb[13], a);
				vsum[14] = vmlaq_n_f32(vsum[14], vb[14], a);
				vsum[15] = vmlaq_n_f32(vsum[15], vb[15], a);
			}
			vsum[ 0] = vmulq_n_f32(vsum[ 0], ALPHA);
			vsum[ 1] = vmulq_n_f32(vsum[ 1], ALPHA);
			vsum[ 2] = vmulq_n_f32(vsum[ 2], ALPHA);
			vsum[ 3] = vmulq_n_f32(vsum[ 3], ALPHA);
			vsum[ 4] = vmulq_n_f32(vsum[ 4], ALPHA);
			vsum[ 5] = vmulq_n_f32(vsum[ 5], ALPHA);
			vsum[ 6] = vmulq_n_f32(vsum[ 6], ALPHA);
			vsum[ 7] = vmulq_n_f32(vsum[ 7], ALPHA);
			vsum[ 8] = vmulq_n_f32(vsum[ 8], ALPHA);
			vsum[ 9] = vmulq_n_f32(vsum[ 9], ALPHA);
			vsum[10] = vmulq_n_f32(vsum[10], ALPHA);
			vsum[11] = vmulq_n_f32(vsum[11], ALPHA);
			vsum[12] = vmulq_n_f32(vsum[12], ALPHA);
			vsum[13] = vmulq_n_f32(vsum[13], ALPHA);
			vsum[14] = vmulq_n_f32(vsum[14], ALPHA);
			vsum[15] = vmulq_n_f32(vsum[15], ALPHA);
			vc[ 0] = vld1q_f32(C + i * R + j + 4 *  0);
			vc[ 1] = vld1q_f32(C + i * R + j + 4 *  1);
			vc[ 2] = vld1q_f32(C + i * R + j + 4 *  2);
			vc[ 3] = vld1q_f32(C + i * R + j + 4 *  3);
			vc[ 4] = vld1q_f32(C + i * R + j + 4 *  4);
			vc[ 5] = vld1q_f32(C + i * R + j + 4 *  5);
			vc[ 6] = vld1q_f32(C + i * R + j + 4 *  6);
			vc[ 7] = vld1q_f32(C + i * R + j + 4 *  7);
			vc[ 8] = vld1q_f32(C + i * R + j + 4 *  8);
			vc[ 9] = vld1q_f32(C + i * R + j + 4 *  9);
			vc[10] = vld1q_f32(C + i * R + j + 4 * 10);
			vc[11] = vld1q_f32(C + i * R + j + 4 * 11);
			vc[12] = vld1q_f32(C + i * R + j + 4 * 12);
			vc[13] = vld1q_f32(C + i * R + j + 4 * 13);
			vc[14] = vld1q_f32(C + i * R + j + 4 * 14);
			vc[15] = vld1q_f32(C + i * R + j + 4 * 15);
			vsum[ 0] = vmlaq_n_f32(vsum[ 0], vc[ 0], BETA);
			vsum[ 1] = vmlaq_n_f32(vsum[ 1], vc[ 1], BETA);
			vsum[ 2] = vmlaq_n_f32(vsum[ 2], vc[ 2], BETA);
			vsum[ 3] = vmlaq_n_f32(vsum[ 3], vc[ 3], BETA);
			vsum[ 4] = vmlaq_n_f32(vsum[ 4], vc[ 4], BETA);
			vsum[ 5] = vmlaq_n_f32(vsum[ 5], vc[ 5], BETA);
			vsum[ 6] = vmlaq_n_f32(vsum[ 6], vc[ 6], BETA);
			vsum[ 7] = vmlaq_n_f32(vsum[ 7], vc[ 7], BETA);
			vsum[ 8] = vmlaq_n_f32(vsum[ 8], vc[ 8], BETA);
			vsum[ 9] = vmlaq_n_f32(vsum[ 9], vc[ 9], BETA);
			vsum[10] = vmlaq_n_f32(vsum[10], vc[10], BETA);
			vsum[11] = vmlaq_n_f32(vsum[11], vc[11], BETA);
			vsum[12] = vmlaq_n_f32(vsum[12], vc[12], BETA);
			vsum[13] = vmlaq_n_f32(vsum[13], vc[13], BETA);
			vsum[14] = vmlaq_n_f32(vsum[14], vc[14], BETA);
			vsum[15] = vmlaq_n_f32(vsum[15], vc[15], BETA);
			vst1q_f32(C + i * R + j + 4 *  0, vsum[ 0]);
			vst1q_f32(C + i * R + j + 4 *  1, vsum[ 1]);
			vst1q_f32(C + i * R + j + 4 *  2, vsum[ 2]);
			vst1q_f32(C + i * R + j + 4 *  3, vsum[ 3]);
			vst1q_f32(C + i * R + j + 4 *  4, vsum[ 4]);
			vst1q_f32(C + i * R + j + 4 *  5, vsum[ 5]);
			vst1q_f32(C + i * R + j + 4 *  6, vsum[ 6]);
			vst1q_f32(C + i * R + j + 4 *  7, vsum[ 7]);
			vst1q_f32(C + i * R + j + 4 *  8, vsum[ 8]);
			vst1q_f32(C + i * R + j + 4 *  9, vsum[ 9]);
			vst1q_f32(C + i * R + j + 4 * 10, vsum[10]);
			vst1q_f32(C + i * R + j + 4 * 11, vsum[11]);
			vst1q_f32(C + i * R + j + 4 * 12, vsum[12]);
			vst1q_f32(C + i * R + j + 4 * 13, vsum[13]);
			vst1q_f32(C + i * R + j + 4 * 14, vsum[14]);
			vst1q_f32(C + i * R + j + 4 * 15, vsum[15]);
		}
	}
}
#endif /* _HAVE_NEON_ */

int main()
{
	const unsigned NREG = 16;
	const unsigned P = 96;
	const unsigned Q = 363;
	const unsigned R = 3072;
	float *A, *B, *C, *C_ref;
#ifdef _HAVE_NEON_
	float *C_neon;
#endif /* _HAVE_NEON_ */
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
#ifdef _HAVE_NEON_
	C_neon = mkl_malloc(P * R * (32 / 8), 4096);
	memcpy(C_neon, C, P * R * (32 / 8));
#endif /* _HAVE_NEON_ */
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

#ifdef _HAVE_NEON_
	printf("CPU with NEON (%d threads): ", omp_get_max_threads()); fflush(stdout);
	gettimeofday(&start, NULL);
	mf_sgemm_neon(A, B, C_neon, P, Q, R, ALPHA, BETA);
	gettimeofday(&end, NULL);
	printf("%g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, (2 * P * Q * R + 3 * P * R) / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

	printf("Minimum absolute error: %g\n", mf_minimum_absolute_error(C_ref, C_neon, P, R));
	printf("Maximum absolute error: %g\n", mf_maximum_absolute_error(C_ref, C_neon, P, R));
	printf("Minimum relative error: %g\n", mf_minimum_relative_error(C_ref, C_neon, P, R));
	printf("Maximum relative error: %g\n", mf_maximum_relative_error(C_ref, C_neon, P, R));

	mkl_free(C_neon);
#endif /* _HAVE_NEON_ */
	mkl_free(C_ref);
	mkl_free(C);
	mkl_free(B);
	mkl_free(A);
	return 0;
}
