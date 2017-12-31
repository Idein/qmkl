/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include "qmkl.h"
#include "local/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

void error_fatal_core(const char *file, const int line, const char *fmt, ...)
{
    va_list ap;

    fprintf(stderr, "%s:%d: ", file, line);

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    qmkl_memory_dump_allocated();

    exit(EXIT_FAILURE);
}

void xerbla_local_core(const int info, const char *fmt, ...)
{
    int len;
    char str[0x100];
    va_list ap;

    va_start(ap, fmt);
    len = vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    xerbla(str, &info, len);
}

void xerbla(const char *strname, const int *info, const int len)
{
    UNUSED(len);

    /* xerbla of Intel MKL prints error message to stdout, but we don't. */
    fprintf(stderr, "QMKL error: Parameter %d was incorrect on entry to %s.\n", *info, strname);
}
