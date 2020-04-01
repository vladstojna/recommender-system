#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void die(const char *error) {
	fprintf(stderr, "Error: %s\n", error);
	exit(-1);
}

void parse_int(FILE *fp, int *i) {
	if (fscanf(fp, "%d", i) != 1) {
		die("Error in int argument.");
	}
}

void parse_double(FILE *fp, double *d) {
	if (fscanf(fp, "%lf", d) != 1) {
		die("Error in double argument.");
	}
}

void parse_three_ints(FILE *fp, int *a, int *b, int *c) {
	if (fscanf(fp, "%d %d %d", a, b, c) != 3) {
		die("Error in multiple int argument.");
	}
}

void parse_non_zero_entry(FILE *fp, int *row, int *col, double *val) {
	if (fscanf(fp, "%d %d %lf", row, col, val) != 3) {
		die("Error in non-zero entry.");
	}
}

void memset_parallel(void *s, int val, size_t sz, size_t type_sz) {
	int num_threads = omp_get_num_threads();
	int id = omp_get_thread_num();
	size_t stride = sz / num_threads + (sz % num_threads != 0);
	memset((char*) s + id * stride * type_sz,
		val,
		type_sz * (stride * (id + 1) > sz ? sz - stride * id : stride));
}

void memcpy_parallel(void *dest, const void *src, size_t sz, size_t type_sz) {
	int num_threads = omp_get_num_threads();
	int id = omp_get_thread_num();
	size_t stride = sz / num_threads + (sz % num_threads != 0);
	memcpy((char*) dest + id * stride * type_sz,
		(char*) src + id * stride * type_sz,
		type_sz * (stride * (id + 1) > sz ? sz - stride * id : stride));
}
