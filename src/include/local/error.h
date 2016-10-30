/*
 * Copyright (c) 2016 Sugizaki Yukimasa (ysugi@idein.jp)
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#ifndef _LOCAL_ERROR_H
#define _LOCAL_ERROR_H

	void error_fatal_core(const char *file, const int line, const char *fmt, ...);

#define error_fatal(fmt, ...) error_fatal_core(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

#endif /* _LOCAL_ERROR_H */
