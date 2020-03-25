#include "util.h"

#include <stdio.h>
#include <stdlib.h>

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