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

void blas_gemm_init()
{
	if (++called.blas_gemm != 1)
		return;

	unif_and_code_size_req(12 * (32 / 8), sizeof(code_sgemm));
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

	memcpy(code_common_cpu, code_sgemm, sizeof(code_sgemm));

	launch_qpu_code_mailbox(1, 1, 100e3, unif_common_gpu, code_common_gpu);
}
