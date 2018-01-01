/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
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
#include <string.h>

static const unsigned code_sabs[] = {
#include "sAbs.qhex"
};

void vm_abs_init()
{
    if (++called.vm_abs != 1)
        return;

    unif_and_code_size_req(3 * (32 / 8), sizeof(code_sabs));
}

void vm_abs_finalize()
{
    if (--called.vm_abs != 0)
        return;
}

void vsAbs(MKL_INT n, const float *a, float *y)
{
    unsigned a_gpu = get_ptr_gpu_from_ptr_cpu(a);
    unsigned y_gpu = get_ptr_gpu_from_ptr_cpu(y);
    unsigned *p;
    const int vector_length = 3 * 16 * 16;

    if (n <= vector_length)
        error_fatal("n must be greater than %d\n", vector_length);
    if (n % vector_length != 0)
        error_fatal("n must be a multiple of %d\n", vector_length);

    p = unif_common_cpu;
    unif_add_uint(n / vector_length - 1, &p);
    unif_add_uint(a_gpu,                 &p);
    unif_add_uint(y_gpu,                 &p);

    memcpy(code_common_cpu, code_sabs, sizeof(code_sabs));

    qmkl_cache_op(a, n * sizeof(*a), QMKL_CACHE_OP_CLEAN);
    qmkl_cache_op(y, n * sizeof(*y), QMKL_CACHE_OP_CLEAN);
    launch_qpu_code_mailbox(1, 0, 5e3, unif_common_gpu, code_common_gpu);
    qmkl_cache_op(a, n * sizeof(*a), QMKL_CACHE_OP_INVALIDATE);
    qmkl_cache_op(y, n * sizeof(*y), QMKL_CACHE_OP_INVALIDATE);
}
