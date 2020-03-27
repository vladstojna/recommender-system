#include <time.h>

#define ENABLE_BENCH

#ifdef ENABLE_BENCH

#define __init_benchmark \
	struct timespec __start_timespec__; \
	struct timespec __finish_timespec__;

#define __start_benchmark \
	clock_gettime(CLOCK_MONOTONIC, &__start_timespec__);

#define __end_benchmark(msg, prec) \
	clock_gettime(CLOCK_MONOTONIC, &__finish_timespec__); \
	if (__finish_timespec__.tv_nsec - __start_timespec__.tv_nsec < 0) { \
		__finish_timespec__.tv_sec = __finish_timespec__.tv_sec - __start_timespec__.tv_sec - 1; \
		__finish_timespec__.tv_nsec = 1000000000 + __finish_timespec__.tv_nsec - __start_timespec__.tv_nsec; \
	} else { \
		__finish_timespec__.tv_sec = __finish_timespec__.tv_sec - __start_timespec__.tv_sec; \
		__finish_timespec__.tv_nsec = __finish_timespec__.tv_nsec - __start_timespec__.tv_nsec; \
	} \
	printf("%s : %f\n", msg, __finish_timespec__.tv_sec * (double) prec + __finish_timespec__.tv_nsec / ((1e9) / (double) prec));

#else

#define __init_benchmark // empty
#define __start_benchmark // empty
#define __end_benchmark(msg, prec) // empty

#endif

/* Usage
__init_benchmark // only once
__start_benchmark
work();
__end_benchmark("benchmark 1", 1e3)
*/
