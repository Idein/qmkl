/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _QMKL_MEMORY_H_
#define _QMKL_MEMORY_H_

#include <sys/types.h>
#include "qmkl/types.h"

    enum qmkl_cache_op {
        QMKL_CACHE_OP_INVALIDATE,
        QMKL_CACHE_OP_CLEAN
    };

    void memory_init();
    void memory_finalize();
    void* map_on_cpu(MKL_UINT ptr_gpu, size_t alloc_size);
    void unmap_on_cpu(void *ptr_cpu, size_t alloc_size);
    void* mkl_malloc_cache(size_t alloc_size, int alignment,
            const _Bool use_cpu_cache);
    void* mkl_malloc(size_t alloc_size, int alignment);
    void mkl_free(void *a_ptr);
    MKL_UINT get_ptr_gpu_from_ptr_cpu(const void *ptr_cpu);
    void unif_set_uint(MKL_UINT *p, const MKL_UINT u);
    void unif_set_float(MKL_UINT *p, const float f);
    void unif_add_uint(const MKL_UINT u, MKL_UINT **p);
    void unif_add_float(const float f, MKL_UINT **p);

    int qmkl_cache_op(const void *p, const size_t size, enum qmkl_cache_op op);

#define BUS_TO_PHYS(addr) ((addr) & ~0xc0000000)

#endif /* _QMKL_MEMORY_H_ */
