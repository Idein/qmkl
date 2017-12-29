/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _QMKL_MAP_H_
#define _QMKL_MAP_H_

#include <sys/types.h>
#include "qmkl/types.h"

    void map_init();
    void map_finalize();
    void* map_on_cpu(MKL_UINT ptr_gpu, size_t alloc_size);
    void unmap_on_cpu(void *ptr_cpu, size_t alloc_size);

#define BUS_TO_PHYS(addr) ((addr) & ~0xc0000000)

#endif /* _QMKL_MAP_H_ */
