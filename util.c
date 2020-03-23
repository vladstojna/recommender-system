#include "util.h"

#include <stdio.h>
#include <stdlib.h>

void die(const char *error) {
	fprintf(stderr, "Error: %s\n", error);
	exit(-1);
}

int parse_int(FILE *fp) {
	int value;

	if (fscanf(fp, "%d", &value) != 1) {
		die("Error in int argument.");
	}

	return value;
}

double parse_double(FILE *fp) {
	double value;

	if (fscanf(fp, "%lf", &value) != 1) {
		die("Error in double argument.");
	}

	return value;
}