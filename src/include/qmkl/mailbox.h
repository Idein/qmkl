/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _QMKL_MAILBOX_H_
#define _QMKL_MAILBOX_H_

#include <stdint.h>

	void mailbox_init();
	void mailbox_finalize();
	int mailbox_open();
	void mailbox_close(int fd_mb);
	void mailbox_property(int fd_mb, void *buf);
	uint32_t mailbox_mem_alloc(int fd_mb, uint32_t size, uint32_t align, uint32_t flags);
	uint32_t mailbox_mem_free(int fd_mb, uint32_t handle);
	uint32_t mailbox_mem_lock(int fd_mb, uint32_t handle);
	uint32_t mailbox_mem_unlock(int fd_mb, uint32_t ptr_gpu);
	uint32_t mailbox_qpu_execute(int fd_mb, uint32_t num_qpus, uint32_t control, uint32_t noflush, uint32_t timeout);
	uint32_t mailbox_qpu_enable(int fd_mb, uint32_t enable);

#define MEM_FLAG_DISCARDABLE      (1 << 0)
#define MEM_FLAG_NORMAL           (0 << 2)
#define MEM_FLAG_DIRECT           (1 << 2)
#define MEM_FLAG_COHERENT         (1 << 3)
#define MEM_FLAG_L1_NONALLOCATING (MEM_FLAG_DIRECT | MEM_FLAG_COHERENT)
#define MEM_FLAG_ZERO             (1 << 4)
#define MEM_FLAG_NO_INIT          (1 << 5)
#define MEM_FLAG_HINT_PERMALOCK   (1 << 6)

#endif /* _QMKL_MAILBOX_H_ */
