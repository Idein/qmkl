/*
 * Copyright (c) 2018 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include <interface/vcsm/user-vcsm.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static double get_time()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double) t.tv_sec + t.tv_usec * 1e-6;
}

static void read_speed(const unsigned * const p, const size_t size)
{
    size_t i;
    double start, end;
    start = get_time();
    for (i = 0; i < size / 4; i ++)
        ((volatile unsigned*) p)[i];
    end = get_time();
    printf("%g [s], %g [B/s]\n", end - start, size / (end - start));
}

static void write_speed(const unsigned *p, const size_t size)
{
    static unsigned pat = 0xdeadbeef;
    size_t i;
    double start, end;
    start = get_time();
    for (i = 0; i < size / 4; i ++)
        ((volatile unsigned*) p)[i] = pat;
    end = get_time();
    for (i = 0; i < size / 4; i ++)
        if (p[i] != pat)
            printf("error at i=%zu\n", i);
    printf("%g [s], %g [B/s]\n", end - start, size / (end - start));
    pat = pat ^ (pat << 13);
    pat = pat ^ (pat >> 17);
}

int main()
{
    int err;
    const size_t size = 65536 * 512;
    unsigned hdl_nc, hdl_c;
    void *p_nc, *p_c;
    int i;

    err = vcsm_init();
    if (err) {
        fprintf(stderr, "Failed to initialize user-vcsm\n");
        return err;
    }

    hdl_nc = vcsm_malloc_cache(size, VCSM_CACHE_TYPE_NONE, "qmkl/test");
    if (!hdl_nc) {
        fprintf(stderr, "Failed to allocate\n");
        return err;
    }
    hdl_c  = vcsm_malloc_cache(size, VCSM_CACHE_TYPE_HOST, "qmkl/test");
    if (!hdl_c) {
        fprintf(stderr, "Failed to allocate\n");
        return err;
    }

    p_nc = vcsm_lock(hdl_nc);
    if (p_nc == NULL) {
        fprintf(stderr, "Failed to lock\n");
        return err;
    }
    p_c  = vcsm_lock(hdl_c);
    if (p_c == NULL) {
        fprintf(stderr, "Failed to lock\n");
        return err;
    }

    printf("Read test. Using size=%g[MB]. Measuring 4 times each.\n",
            (double) size * 1e-6);
    for (i = 0; i < 4; i ++) {
        printf("cached     #%d: ", i);
        read_speed(p_c,  size);
        printf("non-cached #%d: ", i);
        read_speed(p_nc, size);
    }

    printf("\nWrite test. Using size=%g[MB]. Measuring 4 times each.\n",
            (double) size * 1e-6);
    for (i = 0; i < 4; i ++) {
        printf("cached     #%d: ", i);
        write_speed(p_c,  size);
        printf("non-cached #%d: ", i);
        write_speed(p_nc, size);
    }

    err = vcsm_unlock_ptr(p_nc);
    if (err) {
        fprintf(stderr, "Failed to unlock\n");
        return err;
    }
    err = vcsm_unlock_ptr(p_c);
    if (err) {
        fprintf(stderr, "Failed to unlock\n");
        return err;
    }
    vcsm_free(hdl_nc);
    vcsm_free(hdl_c);
    vcsm_exit();
    return 0;
}
