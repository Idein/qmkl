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
	.launch_qpu_code = 0
};

static const int unif_len = 1024;
MKL_UINT *unif_common_cpu = NULL;
MKL_UINT unif_common_gpu = 0;

void qmkl_init()
{
	int ret_int;

	ret_int = atexit(qmkl_finalize);
	if (ret_int != 0)
		error_fatal("atexit returned %d\n", ret_int);

	if (++called.main != 1)
		return;

	mailbox_init();
	memory_init();
	launch_qpu_code_init();

	unif_common_cpu = mkl_malloc(unif_len * (32 / 8), 4096);
	unif_common_gpu = get_ptr_gpu_from_ptr_cpu(unif_common_cpu);
}

void qmkl_finalize()
{
	if (--called.main != 0)
		return;

	mkl_free(unif_common_cpu);

	launch_qpu_code_finalize();
	memory_finalize();
	mailbox_finalize();
}
