/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _QMKL_BLAS_H_
#define _QMKL_BLAS_H_

#include "qmkl/types.h"

#define CBLAS_LAYOUT MKL_UINT
#define CblasRowMajor (1 << 0)
#define CblasColMajor (1 << 1)

#define CBLAS_TRANSPOSE MKL_UINT
#define CblasNoTrans   (1 << 0)
#define CblasTrans     (1 << 1)
#define CblasConjTrans (1 << 2)

    void blas_gemm_init();
    void blas_gemm_finalize();
    void blas_copy_init();
    void blas_copy_finalize();
    void blas_omatcopy_init();
    void blas_omatcopy_finalize();

    void cblas_sgemm(
        const CBLAS_LAYOUT layout,
        const CBLAS_TRANSPOSE transa,
        const CBLAS_TRANSPOSE transb,
        const MKL_INT m,
        const MKL_INT n,
        const MKL_INT k,
        const float alpha,
        const float *a,
        const MKL_INT lda,
        const float *b,
        const MKL_INT ldb,
        const float beta,
        float *c,
        const MKL_INT ldc);

    void cblas_scopy(
        const MKL_INT n,
        const float *x,
        const MKL_INT incx,
        float *y,
        const MKL_INT incy);

    void mkl_somatcopy(
        char ordering, char trans,
        const size_t rows, const size_t cols,
        const float alpha,
        const float *A, const size_t lda,
        float *B, const size_t ldb);

#endif /* _QMKL_BLAS_H_ */
