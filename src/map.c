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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <errno.h>

static MKL_INT fd_dev_mem = -1;
static MKL_LONG pagesize = 0;

void map_init()
{
    if (++called.map != 1)
        return;

    fd_dev_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_dev_mem == -1)
        error_fatal("Failed to open /dev/mem: %s\n", strerror(errno));

    pagesize = sysconf(_SC_PAGESIZE);
}

void map_finalize()
{
    MKL_INT ret_int;

    if (--called.map != 0)
        return;

    ret_int = close(fd_dev_mem);
    if (ret_int == -1)
        error_fatal("Failed to close /dev/mem: %s\n", strerror(errno));
    fd_dev_mem = -1;
}

void* map_on_cpu(MKL_UINT ptr_gpu, size_t alloc_size)
{
    void *ptr_cpu;

    if (ptr_gpu % pagesize != 0)
        error_fatal("GPU memory 0x%08x is not pagesize(%ld)-aligned\n", ptr_gpu, pagesize);

    ptr_cpu = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_dev_mem, ptr_gpu);
    if (ptr_cpu == MAP_FAILED)
        error_fatal("Failed to map GPU memory 0x%08x\n", ptr_gpu);

    return ptr_cpu;
}

void unmap_on_cpu(void *ptr_cpu, size_t alloc_size)
{
    MKL_INT ret_int;

    ret_int = munmap(ptr_cpu, alloc_size);
    if (ret_int != 0)
        error_fatal("Failed to unmap CPU memory %p\n", ptr_cpu);
}
