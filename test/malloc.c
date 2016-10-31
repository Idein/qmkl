#include "qmkl.h"
#include <string.h>

int main()
{
	int *p = NULL;
	int size = 1024 * sizeof(*p);
	qmkl_init();
	p = mkl_malloc(size, 4096);
	memset(p, 0x5a, size);
	memset(p, 0xa5, size);
	mkl_free(p);
	qmkl_finalize();
	return 0;
}
