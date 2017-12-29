/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _QMKL_H_
#define _QMKL_H_

    void qmkl_init() __attribute__((constructor));
    void qmkl_finalize() __attribute__((destructor));

#include "qmkl/types.h"
#include "qmkl/raspberrypi-firmware.h"
#include "qmkl/mailbox.h"
#include "qmkl/memory.h"
#include "qmkl/map.h"
#include "qmkl/launch_qpu_code.h"
#include "qmkl/blas.h"
#include "qmkl/vm.h"
#include "qmkl/error.h"

#endif /* _QMKL_H_ */
