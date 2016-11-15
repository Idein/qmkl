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
#include <stdlib.h>
#include <sys/types.h>

struct called called = {
	.main = 0,
	.mailbox = 0,
	.memory = 0,
	.launch_qpu_code = 0,
	.blas_gemm = 0
};

static size_t unif_size = 0, code_size = 0;
MKL_UINT *unif_common_cpu = NULL, *code_common_cpu = NULL;
MKL_UINT unif_common_gpu = 0, code_common_gpu = 0;

void qmkl_init()
{
	int ret_int;

	if (++called.main != 1)
		return;

	ret_int = atexit(qmkl_finalize);
	if (ret_int != 0)
		error_fatal("atexit returned %d\n", ret_int);

	mailbox_init();
	memory_init();
	launch_qpu_code_init();
	blas_gemm_init();

	if (unif_size != 0) {
		unif_common_cpu = mkl_malloc(unif_size, 4096);
		unif_common_gpu = get_ptr_gpu_from_ptr_cpu(unif_common_cpu);
	}
	if (code_size != 0) {
		code_common_cpu = mkl_malloc(code_size, 4096);
		code_common_gpu = get_ptr_gpu_from_ptr_cpu(code_common_cpu);
	}
}

void qmkl_finalize()
{
	if (--called.main != 0)
		return;

	mkl_free(code_common_cpu);
	mkl_free(unif_common_cpu);

	blas_gemm_finalize();
	launch_qpu_code_finalize();
	memory_finalize();
	mailbox_finalize();
}

void unif_and_code_size_req(const size_t unif_size_req, const size_t code_size_req)
{
	if (unif_size_req > unif_size)
		unif_size = unif_size_req;
	if (code_size_req > code_size)
		code_size = code_size_req;
}
