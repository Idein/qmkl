/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include "qmkl.h"
#include "local/called.h"
#include "local/error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <errno.h>

static MKL_INT fd_mb = -1;
static MKL_INT fd_dev_mem = -1;
static MKL_LONG pagesize = 0;

struct mem_allocated_list {
	size_t alloc_size;
	MKL_UINT handle;
	MKL_UINT ptr_gpu;
	MKL_UINT *ptr_cpu;
	struct mem_allocated_list *next;
} *mem_allocated_list_head = NULL;

void memory_init()
{
	if (++called.memory != 1)
		return;

	mailbox_init();
	fd_mb = mailbox_open();

	fd_dev_mem = open("/dev/mem", O_RDWR | O_SYNC);
	if (fd_dev_mem == -1)
		error_fatal("Failed to open /dev/mem: %s\n", strerror(errno));

	pagesize = sysconf(_SC_PAGESIZE);
}

void memory_finalize()
{
	MKL_INT ret_int;

	if (--called.memory != 0)
		return;

	ret_int = close(fd_dev_mem);
	if (ret_int == -1)
		error_fatal("Failed to close /dev/mem: %s\n", strerror(errno));
	fd_dev_mem = -1;

	mailbox_close(fd_mb);
	mailbox_finalize();
}

void* map_on_cpu(MKL_UINT ptr_gpu, size_t alloc_size)
{
	void *ptr_cpu;

	if (ptr_gpu % pagesize != 0)
		error_fatal("GPU memory 0x%08x is not pagesize(%ld)-aligned\n", ptr_gpu, pagesize);

	ptr_cpu = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_dev_mem, ptr_gpu);
	if (ptr_cpu == MAP_FAILED)
		error_fatal("Failed to map GPU memory 0x%08x\n", ptr_gpu);

	return ptr_cpu;
}

void unmap_on_cpu(void *ptr_cpu, size_t alloc_size)
{
	MKL_INT ret_int;

	ret_int = munmap(ptr_cpu, alloc_size);
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
	MKL_UINT handle;
	MKL_UINT ptr_gpu;
	MKL_UINT *ptr_cpu;

	handle = mailbox_mem_alloc(fd_mb, alloc_size, alignment, MEM_FLAG_DIRECT);
	if (!handle)
		error_fatal("Failed to allocate %zu bytes of memory on GPU\n", alloc_size);

	ptr_gpu = mailbox_mem_lock(fd_mb, handle);
	if (ptr_gpu == 0)
		error_fatal("Failed to lock %zu bytes of memory on GPU\n", alloc_size);

	ptr_cpu = map_on_cpu(BUS_TO_PHYS(ptr_gpu), alloc_size);

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
	cur->ptr_cpu = ptr_cpu;
	cur->next = NULL;

	return ptr_cpu;
}

void mkl_free(void *a_ptr)
{
	struct mem_allocated_list *cur = mem_allocated_list_head, *prev = NULL;

	while (cur != NULL) {
		if (cur->ptr_cpu == a_ptr) {
			size_t alloc_size = cur->alloc_size;
			MKL_UINT handle = cur->handle;
			MKL_UINT ptr_gpu = cur->ptr_gpu;
			MKL_UINT *ptr_cpu = cur->ptr_cpu;
			MKL_UINT ret_uint;

			unmap_on_cpu(ptr_cpu, alloc_size);

			ret_uint = mailbox_mem_unlock(fd_mb, ptr_gpu);
			if (ret_uint != 0)
				error_fatal("Failed to unlock GPU memory 0x%08x\n", ptr_gpu);

			ret_uint = mailbox_mem_free(fd_mb, handle);
			if (ret_uint != 0)
				error_fatal("Failed to free GPU handle %d\n", handle);

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

MKL_UINT get_ptr_gpu_from_ptr_cpu(const MKL_UINT *ptr_cpu)
{
	struct mem_allocated_list *cur;

	for (cur = mem_allocated_list_head; cur != NULL; cur = cur->next)
		if (cur->ptr_cpu == ptr_cpu)
			return cur->ptr_gpu;

	error_fatal("No such ptr_cpu: %p\n", ptr_cpu);
}
