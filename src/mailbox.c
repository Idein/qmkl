/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include "qmkl.h"
#include "local/vcio.h"
#include "local/called.h"
#include "local/error.h"
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#define DEVICE_PREFIX "/dev/"

void mailbox_init()
{
}

void mailbox_finalize()
{
}

int mailbox_open()
{
	int fd_mb;

	fd_mb = open(DEVICE_PREFIX DEVICE_FILE_NAME, O_NONBLOCK);
	if (fd_mb == -1)
		error_fatal("Failed to open %s: %s\n", DEVICE_PREFIX DEVICE_FILE_NAME, strerror(errno));

	return fd_mb;
}

void mailbox_close(int fd_mb)
{
	int ret_int;

	ret_int = close(fd_mb);
	if (ret_int == -1)
		error_fatal("Failed to close %s: %s\n", DEVICE_PREFIX DEVICE_FILE_NAME, strerror(errno));
}

void mailbox_property(int fd_mb, void *buf)
{
	int ret_int;

	ret_int = ioctl(fd_mb, IOCTL_MBOX_PROPERTY, buf);
	if (ret_int == -1)
		error_fatal("ioctl: %s\n", strerror(errno));
}

static void check_error(uint32_t p[])
{
	switch (p[1]) {
		case VCMSG_REQUEST_SUCCESSFUL:
			break;
		case VCMSG_REQUEST_FAILED:
			error_fatal("Mailbox process request failed\n");
			break;
		default:
			error_fatal("Unknown Mailbox return code: 0x%08x\n", p[1]);
	}
}

uint32_t mailbox_mem_alloc(int fd_mb, uint32_t size, uint32_t align, uint32_t flags)
{
	int i = 0;
	uint32_t p[16];

	p[i++] = 0;
	p[i++] = VCMSG_PROCESS_REQUEST;
	p[i++] = VCMSG_SET_ALLOCATE_MEM;
	p[i++] = 3 * 4;
	p[i++] = 3 * 4;
	p[i++] = size;
	p[i++] = align;
	p[i++] = flags;
	p[i++] = VCMSG_PROPERTY_END;
	p[0] = i * sizeof(p[0]);

	mailbox_property(fd_mb, p);
	check_error(p);

	return p[5];
}

uint32_t mailbox_mem_free(int fd_mb, uint32_t handle)
{
	int i = 0;
	uint32_t p[16];

	p[i++] = 0;
	p[i++] = VCMSG_PROCESS_REQUEST;
	p[i++] = VCMSG_SET_RELEASE_MEM;
	p[i++] = 1 * 4;
	p[i++] = 1 * 4;
	p[i++] = handle;
	p[i++] = VCMSG_PROPERTY_END;
	p[0] = i * sizeof(p[0]);

	mailbox_property(fd_mb, p);
	check_error(p);

	return p[5];
}

uint32_t mailbox_mem_lock(int fd_mb, uint32_t handle)
{
	int i = 0;
	uint32_t p[16];

	p[i++] = 0;
	p[i++] = VCMSG_PROCESS_REQUEST;
	p[i++] = VCMSG_SET_LOCK_MEM;
	p[i++] = 1 * 4;
	p[i++] = 1 * 4;
	p[i++] = handle;
	p[i++] = VCMSG_PROPERTY_END;
	p[0] = i * sizeof(p[0]);

	mailbox_property(fd_mb, p);
	check_error(p);

	return p[5];
}

uint32_t mailbox_mem_unlock(int fd_mb, uint32_t ptr_gpu)
{
	int i = 0;
	uint32_t p[16];

	p[i++] = 0;
	p[i++] = VCMSG_PROCESS_REQUEST;
	p[i++] = VCMSG_SET_UNLOCK_MEM;
	p[i++] = 1 * 4;
	p[i++] = 1 * 4;
	p[i++] = ptr_gpu;
	p[i++] = VCMSG_PROPERTY_END;
	p[0] = i * sizeof(p[0]);

	mailbox_property(fd_mb, p);
	check_error(p);

	return p[5];
}
