#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void memcpy_parallel(void *dest, const void *src, size_t sz, size_t type_sz, int tid, int num_threads) {
	size_t stride = sz / num_threads;
	memcpy((char*) dest + tid * stride * type_sz,
		(char*) src + tid * stride * type_sz,
		type_sz * ((tid + 1) == num_threads ? sz - stride * tid : stride));
}

void memset_parallel(void *buffer, int value, size_t sz, size_t type_sz, int tid, int num_threads) {
	size_t stride = sz / num_threads;
	memset((char*) buffer + tid * stride * type_sz,
		value,
		type_sz * ((tid + 1) == num_threads ? sz - stride * tid : stride));
}
