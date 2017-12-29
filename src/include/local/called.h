/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _LOCAL_CALLED_H_
#define _LOCAL_CALLED_H_

    extern struct called {
        int main, mailbox, memory, map, launch_qpu_code, blas_gemm, blas_copy,
                vm_abs;
    } called;

#endif /* _LOCAL_CALLED_H_ */
