/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _QMKL_VM_H_
#define _QMKL_VM_H_

#include "qmkl/types.h"

    void vm_abs_init();
    void vm_abs_finalize();

    void vsAbs(MKL_INT n, const float *a, float *y);

#endif /* _QMKL_VM_H_ */
