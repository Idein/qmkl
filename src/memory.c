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
#include <rpimemmgr.h>
#include <stdio.h>

static struct rpimemmgr mgr;

void memory_init()
{
    int ret;

    if (++called.memory != 1)
        return;

    ret = rpimemmgr_init(&mgr);
    if (ret)
        error_fatal("Failed to initialize rpimemmgr\n");
}

void memory_finalize()
{
    int ret;

    if (--called.memory != 0)
        return;

    ret = rpimemmgr_finalize(&mgr);
    if (ret)
        error_fatal("Failed to finalize rpimemmgr\n");
}

void* mkl_malloc_cache(size_t alloc_size, int alignment,
        const VCSM_CACHE_TYPE_T cache_type)
{
    void *ptr_cpu;
    int ret;

    ret = rpimemmgr_alloc_vcsm(alloc_size, alignment, cache_type,
            &ptr_cpu, NULL, &mgr);
    if (ret)
        error_fatal("Failed to allocate memory with rpimemmgr\n");

    return ptr_cpu;
}

/* WARNING: Cached!!! */
void* mkl_malloc(size_t alloc_size, int alignment)
{
    return mkl_malloc_cache(alloc_size, alignment, VCSM_CACHE_TYPE_HOST);
}

void mkl_free(void *a_ptr)
{
    int ret;

    ret = rpimemmgr_free_by_usraddr(a_ptr, &mgr);
    if (ret)
        error_fatal("Failed to free memory with rpimemmgr\n");
}

uint32_t get_ptr_gpu_from_ptr_cpu(const void * const ptr_cpu)
{
    uint32_t ptr_gpu;

    ptr_gpu = rpimemmgr_usraddr_to_busaddr((void*) ptr_cpu, &mgr);

    if (!ptr_gpu)
        error_fatal("Failed to get ptr_gpu from ptr_cpu\n");

    return ptr_gpu;
}
