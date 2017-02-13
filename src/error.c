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

    exit(EXIT_FAILURE);
}

void xerbla(const char *strname, const int *info, const int len)
{
    UNUSED(len);

    /* xerbla of Intel MKL prints error message to stdout, but we don't. */
    fprintf(stderr, "QMKL error: Parameter %d was incorrect on entry to %s.\n", *info, strname);
}
