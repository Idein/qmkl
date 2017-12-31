/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _LOCAL_COMMON_H_
#define _LOCAL_COMMON_H_

#include "qmkl/types.h"
#include <sys/types.h>
#include <vc4mem.h>

    extern MKL_UINT *unif_common_cpu, *code_common_cpu;
    extern MKL_UINT unif_common_gpu, code_common_gpu;
    extern struct vc4mem_config *vc4mem_cfgp;

    void unif_and_code_size_req(const size_t unif_size_req, const size_t code_size_req);

#define UNUSED(x) ((void) x)

#endif /* _LOCAL_COMMON_H_ */
