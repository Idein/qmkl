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
#include <stdlib.h>

struct called called = {
	.main = 0,
	.mailbox = 0,
	.memory = 0
};

void qmkl_init()
{
	int ret_int;

	ret_int = atexit(qmkl_finalize);
	if (ret_int != 0)
		error_fatal("atexit returned %d\n", ret_int);

	if (++called.main != 1)
		return;

	mailbox_init();
	memory_init();
}

void qmkl_finalize()
{
	if (--called.main != 0)
		return;

	memory_finalize();
	mailbox_finalize();
}
