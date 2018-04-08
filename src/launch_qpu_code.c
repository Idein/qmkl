/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include "qmkl.h"
#include "local/called.h"
#include "local/error.h"
#include <mailbox.h>
#include <stdio.h>
#include <stdarg.h>

static int fd_mb = -1;
static uint32_t *ml_control_cpu = NULL;
static uint32_t ml_control_gpu = 0;

#define MAX_QPUS 4 * 3

void launch_qpu_code_init()
{
    int ret;

    if (++called.launch_qpu_code != 1)
        return;

    memory_init();

    fd_mb = mailbox_open();
    if (fd_mb == -1)
        error_fatal("Failed to open Mailbox\n");

    ret = mailbox_qpu_enable(fd_mb, 1);
    if (ret)
        error_fatal("Failed to enable QPU\n");

    ml_control_cpu = mkl_malloc_cache(MAX_QPUS * 2 * (32 / 8), 4096, 0);
    ml_control_gpu = get_ptr_gpu_from_ptr_cpu(ml_control_cpu);
}

void launch_qpu_code_finalize()
{
    int ret;

    if (--called.launch_qpu_code != 0)
        return;

    mkl_free(ml_control_cpu);

    ret = mailbox_qpu_enable(fd_mb, 0);
    if (ret)
        error_fatal("Failed to disable QPU\n");

    ret = mailbox_close(fd_mb);
    if (ret)
        error_fatal("Failed to close Mailbox\n");

    memory_finalize();
}

void launch_qpu_code_mailbox(uint32_t num_qpus, uint32_t noflush, uint32_t timeout, ...)
{
    unsigned i;
    va_list ap;
    uint32_t ret;

    if (num_qpus > MAX_QPUS)
        error_fatal("Too many QPUs: %d (max:%d)\n", num_qpus, MAX_QPUS);

    va_start(ap, timeout);
    for (i = 0; i < num_qpus; i ++) {
        ml_control_cpu[i * 2 + 0] = va_arg(ap, uint32_t); /* unif addr for QPU i */
        ml_control_cpu[i * 2 + 1] = va_arg(ap, uint32_t); /* prog addr for QPU i */
    }
    va_end(ap);

    ret = mailbox_qpu_execute(fd_mb, num_qpus, ml_control_gpu, noflush, timeout);
    if (ret)
        xerbla_local(ret);
}
