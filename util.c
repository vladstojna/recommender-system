#include "util.h"

#include <stdio.h>
#include <stdlib.h>

void die(const char* error) {
	fprintf(stderr, "Error: %s\n", error);
	exit(-1);
}

int parse_int(FILE* file) {
	int value;

	if (fscanf(file, "%d", &value) != 1) {
		fprintf(stderr, "Error in int argument.\n");
		exit(-1);
	}

	return value;
}

double parse_double(FILE* file) {
	double value;

	if (fscanf(file, "%lf", &value) != 1) {
		fprintf(stderr, "Error in double argument.\n");
		exit(-1);
	}

	return value;
}