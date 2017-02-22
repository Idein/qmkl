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

static const unsigned code_sgemm_RNN[] = {
#include "sgemm_RNN.qhex"
};
static const unsigned code_sgemm_RTT[] = {
#include "sgemm_RTT.qhex"
};

static const int unif_len_1th = 14;


void blas_gemm_init()
{
    if (++called.blas_gemm != 1)
        return;

    unif_and_code_size_req(unif_len_1th * (32 / 8), sizeof(code_sgemm_RNN));
    unif_and_code_size_req(unif_len_1th * (32 / 8), sizeof(code_sgemm_RTT));
}

void blas_gemm_finalize()
{
    if (--called.blas_gemm != 0)
        return;
}

static void cblas_sgemm_RNN(
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

    memcpy(code_common_cpu, code_sgemm_RNN, sizeof(code_sgemm_RNN));
    p = unif_common_cpu;
    {
        unsigned th, i, j;
        for (th = 0; th < n_threads; th ++) {
            unif_set_uint (p + th * unif_len_1th +  0, (unsigned) ((unsigned*) unif_common_gpu + th * unif_len_1th));
            unif_set_uint (p + th * unif_len_1th +  7, lda * (32 / 8));
            unif_set_uint (p + th * unif_len_1th +  8, ldb * (32 / 8));
            unif_set_uint (p + th * unif_len_1th +  9, ldc * (32 / 8));
            unif_set_float(p + th * unif_len_1th + 10, ALPHA);
            unif_set_float(p + th * unif_len_1th + 11, BETA);
            unif_set_uint (p + th * unif_len_1th + 12, th);
            unif_set_uint (p + th * unif_len_1th + 13, n_threads);
        }
        th = 0;
        const unsigned P_up = P / 16;
        const unsigned h = (P_up + p_div - 1) / p_div;
        const unsigned h_len = p_div - (h * p_div - P_up);
        const unsigned R_up = R / 64;
        const unsigned w = (R_up + r_div - 1) / r_div;
        const unsigned w_len = r_div - (w * r_div - R_up);
        unsigned h_acc = 0;
        for (i = 0; i < p_div; i ++) {
            unsigned hi = 0;
            if (i == p_div-1) {
                hi = P - h_acc;
            } else {
                hi = i < h_len ? 16 * h : 16 * (h-1);
            }
            unsigned w_acc = 0;
            for (j = 0; j < r_div; j ++) {
                unsigned wj = 0;
                if (j == r_div-1) {
                    wj = R - w_acc;
                } else {
                    wj = j < w_len ? 64 * w : 64 * (w-1);
                }
                unif_set_uint(p + th * unif_len_1th +  1, hi);
                unif_set_uint(p + th * unif_len_1th +  2, Q);
                unif_set_uint(p + th * unif_len_1th +  3, wj);
                unif_set_uint(p + th * unif_len_1th +  4, (unsigned) ((unsigned*)a_gpu + h_acc * lda        ));
                unif_set_uint(p + th * unif_len_1th +  5, (unsigned) ((unsigned*)b_gpu +               w_acc));
                unif_set_uint(p + th * unif_len_1th +  6, (unsigned) ((unsigned*)c_gpu + h_acc * ldc + w_acc));
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

static void cblas_sgemm_RTT(
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

    unsigned p_div = 6;
    if (P < p_div*64) p_div = 4;
    if (P < p_div*64) p_div = 3;
    if (P < p_div*64) p_div = 2;
    if (P < p_div*64) p_div = 1;

    unsigned r_div = 12 / p_div;
    for (; 2 <= r_div; --r_div) {
        if (R >= r_div*16) break;
    }

    const unsigned n_threads = p_div * r_div;

    memcpy(code_common_cpu, code_sgemm_RTT, sizeof(code_sgemm_RTT));
    p = unif_common_cpu;
    {
        unsigned th, i, j;
        for (th = 0; th < n_threads; th ++) {
            unif_set_uint (p + th * unif_len_1th +  0, (unsigned) ((unsigned*) unif_common_gpu + th * unif_len_1th));
            unif_set_uint (p + th * unif_len_1th +  7, lda * (32 / 8));
            unif_set_uint (p + th * unif_len_1th +  8, ldb * (32 / 8));
            unif_set_uint (p + th * unif_len_1th +  9, ldc * (32 / 8));
            unif_set_float(p + th * unif_len_1th + 10, ALPHA);
            unif_set_float(p + th * unif_len_1th + 11, BETA);
            unif_set_uint (p + th * unif_len_1th + 12, th);
            unif_set_uint (p + th * unif_len_1th + 13, n_threads);
        }
        th = 0;
        const unsigned P_up = P / 64;
        const unsigned h = (P_up + p_div - 1) / p_div;
        const unsigned h_len = p_div - (h * p_div - P_up);
        const unsigned R_up = R / 16;
        const unsigned w = (R_up + r_div - 1) / r_div;
        const unsigned w_len = r_div - (w * r_div - R_up);
        unsigned h_acc = 0;
        for (i = 0; i < p_div; i ++) {
            unsigned hi = 0;
            if (i == p_div-1) {
                hi = P - h_acc;
            } else {
                hi = i < h_len ? 64 * h : 64 * (h-1);
            }
            unsigned w_acc = 0;
            for (j = 0; j < r_div; j ++) {
                unsigned wj = 0;
                if (j == r_div-1) {
                    wj = R - w_acc;
                } else {
                    wj = j < w_len ? 16 * w : 16 * (w-1);
                }
                unif_set_uint(p + th * unif_len_1th +  1, hi);
                unif_set_uint(p + th * unif_len_1th +  2, Q);
                unif_set_uint(p + th * unif_len_1th +  3, wj);
                unif_set_uint(p + th * unif_len_1th +  4, (unsigned) ((unsigned*)a_gpu +               h_acc));
                unif_set_uint(p + th * unif_len_1th +  5, (unsigned) ((unsigned*)b_gpu + w_acc * ldb        ));
                unif_set_uint(p + th * unif_len_1th +  6, (unsigned) ((unsigned*)c_gpu + h_acc * ldc + w_acc));
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

static void cblas_sgemm_R(
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
    if (CblasNoTrans == transa) {
        if (CblasNoTrans == transb) {
            return cblas_sgemm_RNN(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } else {
            error_fatal("transa and transb must equal for now\n");
        }
    } else {
        if (CblasNoTrans == transb) {
            error_fatal("transa and transb must equal for now\n");
        } else {
            return cblas_sgemm_RTT(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    }
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
    switch (layout) {
    case CblasColMajor: {
        error_fatal("layout must be RowMajor for now\n");
    } break;
    case CblasRowMajor: {
        return cblas_sgemm_R(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    }
}
