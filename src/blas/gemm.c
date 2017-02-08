/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include "qmkl.h"
#include "local/common.h"
#include "local/called.h"
#include "local/error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned code_sgemm[] = {
#include "sgemm.qhex"
};
static const unsigned code_sgemm_1thread[] = {
#include "sgemm_1thread.qhex"
};


void blas_gemm_init()
{
	if (++called.blas_gemm != 1)
		return;

	unif_and_code_size_req(12 * (32 / 8), sizeof(code_sgemm_1thread));
	unif_and_code_size_req(13 * (32 / 8), sizeof(code_sgemm));
}

void blas_gemm_finalize()
{
	if (--called.blas_gemm != 0)
		return;
}

void cblas_sgemm(
	const CBLAS_LAYOUT layout,
	const CBLAS_TRANSPOSE transa,
	const CBLAS_TRANSPOSE transb,
	const MKL_INT m,
	const MKL_INT n,
	const MKL_INT k,
	const float alpha,
	const float *a,
	const MKL_INT lda,
	const float *b,
	const MKL_INT ldb,
	const float beta,
	float *c,
	const MKL_INT ldc)
{
	MKL_UINT a_gpu = get_ptr_gpu_from_ptr_cpu(a);
	MKL_UINT b_gpu = get_ptr_gpu_from_ptr_cpu(b);
	MKL_UINT c_gpu = get_ptr_gpu_from_ptr_cpu(c);
	uint32_t *p = NULL;

	const unsigned NREG = 64;
	const unsigned P = m;
	const unsigned Q = k;
	const unsigned R = n;
	const float ALPHA = alpha;
	const float BETA = beta;

	if (layout != CblasRowMajor)
		error_fatal("layout must be RowMajor for now\n");
	if (transa != CblasNoTrans || transb != CblasNoTrans)
		error_fatal("trans must be NoTrans for now\n");
	if (lda != k)
		error_fatal("lda must be k for now\n");
	if (ldb != n)
		error_fatal("ldb must be n for now\n");
	if (ldc != n)
		error_fatal("ldc must be n for now\n");

	if (P % 16 != 0)
		error_fatal("P must be a multiple of 16\n");

	if (R % NREG != 0)
		error_fatal("R must be a multiple of NREG\n");

#if 0 /* sgemm_1thread */

	memcpy(code_common_cpu, code_sgemm_1thread, sizeof(code_sgemm_1thread));

	p = unif_common_cpu;
	unif_add_uint (unif_common_gpu, &p);
	unif_add_uint (P / 16,          &p);
	unif_add_uint (Q,               &p);
	unif_add_uint (R / NREG,        &p);
	unif_add_uint (a_gpu,           &p);
	unif_add_uint (b_gpu,           &p);
	unif_add_uint (c_gpu,           &p);
	unif_add_uint (Q * (32 / 8),    &p);
	unif_add_uint (R * (32 / 8),    &p);
	unif_add_uint (R * (32 / 8),    &p);
	unif_add_float(ALPHA,           &p);
	unif_add_float(BETA,            &p);
	launch_qpu_code_mailbox(1, 1, 100e3, unif_common_gpu, code_common_gpu);

#elif 1 /* sgemm (12 threads) */

	memcpy(code_common_cpu, code_sgemm, sizeof(code_sgemm));

	p = unif_common_cpu;
	{
		const unsigned p_div = 2, r_div = 6;
		const unsigned h = P / (16*p_div), w = R / (64*r_div);
		unsigned th, i, j;
		for (th = 0; th < 12; th ++) {
			p[th * 13 +  0] = (unsigned) ((unsigned*) unif_common_gpu + th * 13);
			p[th * 13 +  7] = 1452;
			p[th * 13 +  8] = 12288;
			p[th * 13 +  9] = 12288;
			memcpy(p + th * 13 + 10, &ALPHA, sizeof(float));
			memcpy(p + th * 13 + 11, &BETA, sizeof(float));
			p[th * 13 + 12] = th;
		}
		th = 0;
		for (i = 0; i < p_div; i ++) {
			for (j = 0; j < r_div; j ++) {
				p[th * 13 +  1] = (i != p_div - 1) ? h : (P / 16 - i * h);
				p[th * 13 +  2] = Q;
				p[th * 13 +  3] = (j != r_div - 1) ? w : (R / 64 - j * w);
				p[th * 13 +  4] = (unsigned) ((unsigned*)a_gpu + i * 16 * h * k             );
				p[th * 13 +  5] = (unsigned) ((unsigned*)b_gpu +                  j * 64 * w);
				p[th * 13 +  6] = (unsigned) ((unsigned*)c_gpu + i * 16 * h * n + j * 64 * w);
				th ++;
			}
		}
	}

	launch_qpu_code_mailbox(12, 1, 5e3,
		(unsigned*) unif_common_gpu +  0 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu +  1 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu +  2 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu +  3 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu +  4 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu +  5 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu +  6 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu +  7 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu +  8 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu +  9 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu + 10 * 13, code_common_gpu,
		(unsigned*) unif_common_gpu + 11 * 13, code_common_gpu
	);
#else
#endif
}
