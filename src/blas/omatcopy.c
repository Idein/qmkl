/*
 * Copyright (c) 2017 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include "qmkl.h"
#include "local/common.h"
#include "local/called.h"
#include "local/error.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

static const unsigned code_somatcopy[] = {
#include "somatcopy.qhex"
};

void blas_omatcopy_init()
{
    if (++called.blas_omatcopy != 1)
        return;

    unif_and_code_size_req(6 * (32 / 8), sizeof(code_somatcopy));
}

void blas_omatcopy_finalize()
{
    if (--called.blas_omatcopy != 0)
        return;
}

void mkl_somatcopy(
    char ordering, char trans,
    const size_t rows, const size_t cols,
    const float alpha,
    const float *A, const size_t lda,
    float *B, const size_t ldb)
{
    const unsigned A_gpu = get_ptr_gpu_from_ptr_cpu(A);
    const unsigned B_gpu = get_ptr_gpu_from_ptr_cpu(B);
    unsigned *p = NULL;
    uint32_t load_setup_0, load_setup_16, store_setup_0, store_setup_16;
    uint32_t load_stride_setup, store_stride_setup;
    _Bool store_is_horiz;

    ordering = tolower(ordering);
    trans    = tolower(trans);

    if (ordering == 'r') {
        if (trans == 't' && ldb < rows)
            xerbla_local(ldb);
        if (trans == 'n' && ldb < cols)
            xerbla_local(ldb);
    } else if (ordering == 'c') {
        if (trans == 't' && ldb < cols)
            xerbla_local(ldb);
        if (trans == 'n' && ldb < rows)
            xerbla_local(ldb);
    }

    /* Scaling is not supported for now. */
    if (alpha != 1.0)
        xerbla_local(alpha);

    if (rows % 16 != 0)
        xerbla_local(rows);
    if ((cols - 16) % 32 != 0)
        xerbla_local(cols);

    /* VPM DMA load/store strides are 13-bit fields. */
    if (lda & ~((1 << 13) - 1))
        xerbla_local(lda);
    if (ldb < rows)
        xerbla_local(ldb);
    else if ((ldb - rows) & ~((1 << 13) - 1))
        xerbla_local(ldb);

    store_is_horiz = !(trans == 't');
    load_setup_0 =
           1 << 31 /* DMA load setup */
        |  0 << 28 /* width = 32bit */
        |  0 << 24 /* mpitch = Take from mpitchb */
        |  0 << 20 /* rowlen = 16 */
        |  0 << 16 /* nrows = 16 */
        |  1 << 12 /* vpitch = 1 */
        |  0 << 11 /* 0:horiz 1:vert */
        |  0 <<  4 /* addr_y = 0 */
        |  0;      /* addr_x = 0 */
    load_setup_16 =
           1 << 31 /* DMA load setup */
        |  0 << 28 /* width = 32bit */
        |  0 << 24 /* mpitch = Take from mpitchb */
        |  0 << 20 /* rowlen = 16 */
        |  0 << 16 /* nrows = 16 */
        |  1 << 12 /* vpitch = 1 */
        |  0 << 11 /* 0:horiz 1:vert */
        | 16 <<  4 /* addr_y = 16 */
        |  0;      /* addr_x = 0 */
    store_setup_0 =
           2             << 30 /* DMA store setup */
        | 16             << 23 /* units = 16 */
        | 16             << 16 /* depth = 16 */
        | store_is_horiz << 14 /* 0:vert 1:horiz */
        |  0             <<  7 /* vpmbase_y = 0 */
        |  0             <<  3 /* vpmbase_x = 0 */
        |  0;                  /* width = 32bit */
    store_setup_16 =
           2             << 30 /* DMA store setup */
        | 16             << 23 /* units = 16 */
        | 16             << 16 /* depth = 16 */
        | store_is_horiz << 14 /* 0:vert 1:horiz */
        | 16             <<  7 /* vpmbase_y = 16 */
        |  0             <<  3 /* vpmbase_x = 0 */
        |  0;                  /* width = 32bit */
    load_stride_setup =
           9 << 28
        | (4 * lda);
    store_stride_setup =
           3 << 30
        | (4 * ldb - 4 * 16);

    p = unif_common_cpu;
    unif_add_uint(rows / 16,  &p);
    unif_add_uint((cols - 16) / 32,  &p);
    unif_add_uint(A_gpu,      &p);
    unif_add_uint(B_gpu,      &p);
    unif_add_uint(4 * 16,     &p);
    unif_add_uint(4 * lda * 16, &p);
    unif_add_uint(4 * 16,     &p);
    unif_add_uint(4 * ldb * 16, &p);
    unif_add_uint(load_setup_0,        &p);
    unif_add_uint(load_setup_16,        &p);
    unif_add_uint(store_setup_0,        &p);
    unif_add_uint(store_setup_16,        &p);
    unif_add_uint(load_stride_setup, &p);
    unif_add_uint(store_stride_setup, &p);

    memcpy(code_common_cpu, code_somatcopy, sizeof(code_somatcopy));

    launch_qpu_code_mailbox(1, 0, 5e3, unif_common_gpu, code_common_gpu);
}
