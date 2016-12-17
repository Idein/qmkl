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

static const unsigned code_scopy[] = {
#include "scopy.qhex"
};

void blas_copy_init()
{
	if (++called.blas_copy != 1)
		return;

	unif_and_code_size_req(3 * (32 / 8), sizeof(code_scopy));
}

void blas_copy_finalize()
{
	if (--called.blas_copy != 0)
		return;
}

void cblas_scopy(
	const MKL_INT n,
	const float *x,
	const MKL_INT incx,
	float *y,
	const MKL_INT incy)
{
	MKL_UINT x_gpu = get_ptr_gpu_from_ptr_cpu(x);
	MKL_UINT y_gpu = get_ptr_gpu_from_ptr_cpu(y);
	uint32_t *p = NULL;

	if (incx != 1)
		error_fatal("incx must be 1 for now\n");
	if (incy != 1)
		error_fatal("incy must be 1 for now\n");
	if (n % 4096 != 0)
		error_fatal("n must be a multiple of 4096\n");
	if (n < 8192)
		error_fatal("n must be greater than 8192\n");

	p = unif_common_cpu;
	unif_add_uint(n / (4096 / 4) - 1, &p);
	unif_add_uint(x_gpu,        &p);
	unif_add_uint(y_gpu,        &p);

	memcpy(code_common_cpu, code_scopy, sizeof(code_scopy));

	launch_qpu_code_mailbox(1, 1, 100e3, unif_common_gpu, code_common_gpu);
}
