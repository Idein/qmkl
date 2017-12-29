/*
 * Copyright (c) 2016 Idein Inc. ( http://idein.jp/ )
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include "qmkl.h"
#include "local/called.h"
#include "local/error.h"
#include <interface/vcsm/user-vcsm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#define ALIGN_UP(x, align) \
        (((((unsigned long) ((x))) + ((align)) - 1) & ~(((align)) - 1)))

struct mem_allocated_list {
    size_t alloc_size;
    MKL_UINT handle;
    MKL_UINT ptr_gpu;
    void *ptr_cpu_before_align;
    void *ptr_cpu;
    struct mem_allocated_list *next;
} *mem_allocated_list_head = NULL;

void memory_init()
{
    int ret;

    if (++called.memory != 1)
        return;

    ret = vcsm_init();
    if (ret)
        error_fatal("Failed to initialize vcsm: %d\n", ret);
}

void memory_finalize()
{
    if (--called.memory != 0)
        return;

    vcsm_exit();
}

static struct mem_allocated_list* mem_allocated_list_alloc()
{
    struct mem_allocated_list *p;

    p = malloc(sizeof(*p));
    if (p == NULL)
        error_fatal("Failed to allocate memory for struct mem_allocated_list\n");

    return p;
}

static void mem_allocated_list_free(struct mem_allocated_list *p)
{
    free(p);
}

/*
 * The original mkl_malloc returns NULL on failure,
 * but we exit with error in such a situation
 * because we identify GPU handle and memory
 * according to ptr_cpu.
 */
void* mkl_malloc(size_t alloc_size, int alignment)
{
    struct mem_allocated_list *cur = NULL;
    MKL_UINT handle;
    MKL_UINT ptr_gpu;
    void *ptr_cpu_before_align;
    void *ptr_cpu;

    handle = vcsm_malloc_cache(alloc_size + alignment - 1,
            VCSM_CACHE_TYPE_NONE, "qmkl");
    if (!handle)
        error_fatal("Failed to allocate %zu bytes of memory on GPU\n",
                alloc_size + alignment - 1);

    ptr_cpu_before_align = vcsm_lock(handle);
    if (ptr_cpu_before_align == NULL)
        error_fatal("Failed to lock %zu bytes of memory on GPU\n", alloc_size);

    ptr_cpu = (void*) ALIGN_UP(ptr_cpu_before_align, alignment);

    ptr_gpu = vcsm_vc_addr_from_hdl(handle);
    if (ptr_gpu == 0)
        error_fatal("Failed to get bus address\n");

    if (mem_allocated_list_head == NULL) {
        cur = mem_allocated_list_head = mem_allocated_list_alloc();
    } else {
        struct mem_allocated_list *p = mem_allocated_list_head;
        while (p->next != NULL)
            p = p->next;
        cur = p->next = mem_allocated_list_alloc();
    }

    cur->alloc_size = alloc_size;
    cur->handle = handle;
    cur->ptr_gpu = ptr_gpu;
    cur->ptr_cpu_before_align = ptr_cpu_before_align;
    cur->ptr_cpu = ptr_cpu;
    cur->next = NULL;

    return ptr_cpu;
}

void mkl_free(void *a_ptr)
{
    struct mem_allocated_list *cur = mem_allocated_list_head, *prev = NULL;

    while (cur != NULL) {
        if (cur->ptr_cpu == a_ptr) {
            MKL_UINT handle = cur->handle;
            MKL_UINT ptr_gpu = cur->ptr_gpu;
            MKL_UINT *ptr_cpu_before_align = cur->ptr_cpu_before_align;
            MKL_INT ret_int;

            ret_int = vcsm_unlock_ptr(ptr_cpu_before_align);
            if (ret_int != 0)
                error_fatal("Failed to unlock GPU memory 0x%08x\n", ptr_gpu);

            vcsm_free(handle);

            if (prev == NULL) /* This node is the head. */
                mem_allocated_list_head = mem_allocated_list_head->next;
            else
                prev->next = cur->next;

            mem_allocated_list_free(cur);

            return;
        }
        prev = cur;
        cur = cur->next;
    }

    error_fatal("No such allocated memory: %p\n", a_ptr);
}

MKL_UINT get_ptr_gpu_from_ptr_cpu(const void *ptr_cpu)
{
    struct mem_allocated_list *cur;

    for (cur = mem_allocated_list_head; cur != NULL; cur = cur->next)
        if (cur->ptr_cpu <= ptr_cpu && ptr_cpu < cur->ptr_cpu + cur->alloc_size)
            return cur->ptr_gpu + (ptr_cpu - cur->ptr_cpu);

    error_fatal("No such ptr_cpu: %p\n", ptr_cpu);
}

void unif_set_uint(MKL_UINT *p, const MKL_UINT u)
{
    memcpy(p, &u, sizeof(u));
}

void unif_set_float(MKL_UINT *p, const float f)
{
    memcpy(p, &f, sizeof(f));
}

void unif_add_uint(const MKL_UINT u, MKL_UINT **p)
{
    unif_set_uint((*p)++, u);
}

void unif_add_float(const float f, MKL_UINT **p)
{
    unif_set_float((*p)++, f);
}
