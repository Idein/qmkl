/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
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
#include <stdlib.h>
#include <string.h>

static const unsigned code_sgemm[] = {
#include "sgemm.qhex"
};
static const unsigned code_sgemm_1thread[] = {
#include "sgemm_1thread.qhex"
};

static const int unif_len_1th = 14;


void blas_gemm_init()
{
    if (++called.blas_gemm != 1)
        return;

    unif_and_code_size_req(12 * (32 / 8), sizeof(code_sgemm_1thread));
    unif_and_code_size_req(unif_len_1th * (32 / 8), sizeof(code_sgemm));
}

void blas_gemm_finalize()
{
    if (--called.blas_gemm != 0)
        return;
}

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
    const MKL_INT ldc)
{
    MKL_UINT a_gpu = get_ptr_gpu_from_ptr_cpu(a);
    MKL_UINT b_gpu = get_ptr_gpu_from_ptr_cpu(b);
    MKL_UINT c_gpu = get_ptr_gpu_from_ptr_cpu(c);
    uint32_t *p = NULL;

    const unsigned P = m;
    const unsigned Q = k;
    const unsigned R = n;
    const float ALPHA = alpha;
    const float BETA = beta;

    if (layout != CblasRowMajor)
        error_fatal("layout must be RowMajor for now\n");
    if (transa != CblasNoTrans || transb != CblasNoTrans)
        error_fatal("trans must be NoTrans for now\n");
    if (lda != k)
        error_fatal("lda must be k for now\n");
    if (ldb != n)
        error_fatal("ldb must be n for now\n");
    if (ldc != n)
        error_fatal("ldc must be n for now\n");

    if (P % 16 != 0)
        error_fatal("P must be a multiple of 16\n");

    if (R % 64 != 0)
        error_fatal("R must be a multiple of 64\n");

    unsigned r_div = 6;
    if (R < r_div*64) r_div = 4;
    if (R < r_div*64) r_div = 3;
    if (R < r_div*64) r_div = 2;
    if (R < r_div*64) r_div = 1;

    unsigned p_div = 12 / r_div;
    for (; 2 <= p_div; --p_div) {
        if (P >= p_div*16) break;
    }

    const unsigned n_threads = p_div * r_div;

    if (n_threads == 1) {
        /* single threaded sgemm */

        memcpy(code_common_cpu, code_sgemm_1thread, sizeof(code_sgemm_1thread));

        p = unif_common_cpu;
        unif_add_uint (unif_common_gpu, &p);
        unif_add_uint (P / 16,          &p);
        unif_add_uint (Q,               &p);
        unif_add_uint (R / 64,          &p);
        unif_add_uint (a_gpu,           &p);
        unif_add_uint (b_gpu,           &p);
        unif_add_uint (c_gpu,           &p);
        unif_add_uint (Q * (32 / 8),    &p);
        unif_add_uint (R * (32 / 8),    &p);
        unif_add_uint (R * (32 / 8),    &p);
        unif_add_float(ALPHA,           &p);
        unif_add_float(BETA,            &p);
        launch_qpu_code_mailbox(1, 1, 100e3, unif_common_gpu, code_common_gpu);

    } else {
        /* multi threads sgemm */

        memcpy(code_common_cpu, code_sgemm, sizeof(code_sgemm));
        p = unif_common_cpu;
        {
            unsigned th, i, j;
            for (th = 0; th < n_threads; th ++) {
                unif_set_uint (p + th * unif_len_1th +  0, (unsigned) ((unsigned*) unif_common_gpu + th * unif_len_1th));
                unif_set_uint (p + th * unif_len_1th +  7, Q * (32 / 8));
                unif_set_uint (p + th * unif_len_1th +  8, R * (32 / 8));
                unif_set_uint (p + th * unif_len_1th +  9, R * (32 / 8));
                unif_set_float(p + th * unif_len_1th + 10, ALPHA);
                unif_set_float(p + th * unif_len_1th + 11, BETA);
                unif_set_uint (p + th * unif_len_1th + 12, th);
                unif_set_uint (p + th * unif_len_1th + 13, n_threads);
            }
            th = 0;
            const unsigned h = (P + 16 * p_div - 1) / (16 * p_div);
            const unsigned h_len = p_div - (16 * h * p_div - P) / 16;
            const unsigned w = (R + 64 * r_div - 1) / (64 * r_div);
            const unsigned w_len = r_div - (64 * w * r_div - R) / 64;
            unsigned h_acc = 0;
            for (i = 0; i < p_div; i ++) {
                unsigned hi = i < h_len ? h : h-1;
                unsigned w_acc = 0;
                for (j = 0; j < r_div; j ++) {
                    unsigned wj = j < w_len ? w : w-1;
                    unif_set_uint(p + th * unif_len_1th +  1, hi);
                    unif_set_uint(p + th * unif_len_1th +  2, Q);
                    unif_set_uint(p + th * unif_len_1th +  3, wj);
                    unif_set_uint(p + th * unif_len_1th +  4, (unsigned) ((unsigned*)a_gpu + 16 * h_acc * k             ));
                    unif_set_uint(p + th * unif_len_1th +  5, (unsigned) ((unsigned*)b_gpu +                  64 * w_acc));
                    unif_set_uint(p + th * unif_len_1th +  6, (unsigned) ((unsigned*)c_gpu + 16 * h_acc * n + 64 * w_acc));
                    th ++;
                    w_acc += wj;
                }
                h_acc += hi;
            }
        }
        launch_qpu_code_mailbox(n_threads, 1, 5e3,
                                (unsigned*) unif_common_gpu +  0 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu +  1 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu +  2 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu +  3 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu +  4 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu +  5 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu +  6 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu +  7 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu +  8 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu +  9 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu + 10 * unif_len_1th, code_common_gpu,
                                (unsigned*) unif_common_gpu + 11 * unif_len_1th, code_common_gpu
        );
    }
}
