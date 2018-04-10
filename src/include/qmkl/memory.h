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

#include <interface/vcsm/user-vcsm.h>
#include <sys/types.h>
#include "qmkl/types.h"

    enum qmkl_cache_op {
        QMKL_CACHE_OP_INVALIDATE,
        QMKL_CACHE_OP_CLEAN
    };

    void memory_init();
    void memory_finalize();
    void* mkl_malloc_cache(size_t alloc_size, int alignment,
            const VCSM_CACHE_TYPE_T cache_type);
    void* mkl_malloc(size_t alloc_size, int alignment);
    void mkl_free(void *a_ptr);
    uint32_t get_ptr_gpu_from_ptr_cpu(const void * const ptr_cpu);

#endif /* _QMKL_MEMORY_H_ */
