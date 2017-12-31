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
#include <vc4mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALIGN_UP(x, align) \
        (((((unsigned long) ((x))) + ((align)) - 1) & ~(((align)) - 1)))

struct vc4mem_config *vc4mem_cfgp = NULL;
static struct vc4mem_config vc4mem_cfg_noncached, vc4mem_cfg_cached;

struct mem_allocated_list {
    size_t alloc_size;
    MKL_UINT ptr_gpu;
    void *ptr_cpu_orig;
    void *ptr_cpu;
    struct mem_allocated_list *next;
} *mem_allocated_list_head = NULL;

void qmkl_memory_cache_on_cpu(const int is_cache_on_cpu)
{
    if (is_cache_on_cpu)
        vc4mem_cfgp = &vc4mem_cfg_cached;
    else
        vc4mem_cfgp = &vc4mem_cfg_noncached;
}

void memory_init()
{
    MKL_INT ret_int;

    if (++called.memory != 1)
        return;

    vc4mem_clear_config(&vc4mem_cfg_noncached);
    vc4mem_cfg_noncached.is_noncached = 1;
    ret_int = vc4mem_init(&vc4mem_cfg_noncached);
    if (ret_int)
        error_fatal("Failed to init noncached vc4mem\n");

    vc4mem_clear_config(&vc4mem_cfg_cached);
    vc4mem_cfg_cached.is_noncached = 0;
    ret_int = vc4mem_init(&vc4mem_cfg_cached);
    if (ret_int)
        error_fatal("Failed to init cached vc4mem\n");

    qmkl_memory_cache_on_cpu(0);
}

void memory_finalize()
{
    MKL_INT ret_int;

    if (--called.memory != 0)
        return;

    ret_int = vc4mem_finalize(&vc4mem_cfg_cached);
    if (ret_int)
        error_fatal("Failed to fin cached vc4mem\n");

    ret_int = vc4mem_finalize(&vc4mem_cfg_noncached);
    if (ret_int)
        error_fatal("Failed to fin noncached vc4mem\n");
}

void* map_on_cpu(MKL_UINT ptr_gpu, size_t alloc_size)
{
    void *ptr_cpu;

    ptr_cpu = vc4mem_map_phys_to_user(vc4mem_bus_to_phys(ptr_gpu), alloc_size,
            vc4mem_cfgp);
    if (ptr_cpu == NULL)
        error_fatal("Failed to map GPU memory: 0x%08x\n", ptr_gpu);

    return ptr_cpu;
}

void unmap_on_cpu(void *ptr_cpu, size_t alloc_size)
{
    MKL_INT ret_int;

    ret_int = vc4mem_unmap_phys_to_user(ptr_cpu, alloc_size, vc4mem_cfgp);
    if (ret_int != 0)
        error_fatal("Failed to unmap CPU memory %p\n", ptr_cpu);
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
    MKL_UINT ptr_gpu;
    void *ptr_cpu_orig, *ptr_cpu;

    ptr_cpu_orig = vc4mem_malloc(alloc_size + alignment - 1, &ptr_gpu,
            vc4mem_cfgp);
    if (ptr_cpu_orig == NULL)
        error_fatal("Failed to allocate %zu bytes of memory on GPU\n",
                alloc_size + alignment - 1);

    ptr_cpu = (void*) ALIGN_UP(ptr_cpu_orig, alignment);

    if (mem_allocated_list_head == NULL) {
        cur = mem_allocated_list_head = mem_allocated_list_alloc();
    } else {
        struct mem_allocated_list *p = mem_allocated_list_head;
        while (p->next != NULL)
            p = p->next;
        cur = p->next = mem_allocated_list_alloc();
    }

    cur->alloc_size = alloc_size;
    cur->ptr_gpu = ptr_gpu;
    cur->ptr_cpu_orig = ptr_cpu_orig;
    cur->ptr_cpu = ptr_cpu;
    cur->next = NULL;

    return ptr_cpu;
}

void* mkl_malloc_noncached(size_t alloc_size, int alignment,
        MKL_UINT *ptr_gpup)
{
    void *ptr_cpu;

    qmkl_memory_cache_on_cpu(0);

    ptr_cpu = mkl_malloc(alloc_size, alignment);
    *ptr_gpup = get_ptr_gpu_from_ptr_cpu(ptr_cpu);

    return ptr_cpu;
}

void mkl_free(void *a_ptr)
{
    struct mem_allocated_list *cur = mem_allocated_list_head, *prev = NULL;

    while (cur != NULL) {
        if (cur->ptr_cpu == a_ptr) {
            size_t alloc_size = cur->alloc_size;
            MKL_UINT ptr_gpu = cur->ptr_gpu;
            MKL_UINT *ptr_cpu_orig = cur->ptr_cpu_orig;
            MKL_INT ret_int;

            ret_int = vc4mem_free(ptr_cpu_orig, ptr_gpu, alloc_size,
                    vc4mem_cfgp);
            if (ret_int)
                error_fatal("Failed to free GPU memory: 0x%08x\n", ptr_gpu);

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

void mkl_free_noncached(void *a_ptr)
{
    mkl_free(a_ptr);
}

void qmkl_memory_dump_allocated()
{
    struct mem_allocated_list *cur = mem_allocated_list_head;

    printf("Allocated (but not freed yet) memory areas:\n");
    for (cur = mem_allocated_list_head; cur != NULL; cur = cur->next)
        printf("  cpu=%p gpu=0x%08x size=0x%08x\n", cur->ptr_cpu_orig,
                cur->ptr_gpu, cur->alloc_size);
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
