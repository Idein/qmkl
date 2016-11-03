/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _LAUNCH_QPU_CODE_H_
#define _LAUNCH_QPU_CODE_H_

#include <stdint.h>

	void launch_qpu_code_init();
	void launch_qpu_code_finalize();
	void launch_qpu_code_mailbox(uint32_t num_qpus, uint32_t noflush, uint32_t timeout, ...);

#endif /* _LAUNCH_QPU_CODE_H_ */
