// serial implementation
#include "util.h"
#include "mat2.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "Run ./matFact.out file");
		die("Missing input file name.");
	}

	const char* file_name = argv[1];
	FILE *fp = fopen(file_name, "r");

	if (fp == NULL)
		die("Unable to open input file.\n");

	// Reading number of iterations
	int iters = parse_int(fp);

	// Reading learning rate
	double alpha = parse_double(fp);

	// Reading number of features
	int features = parse_int(fp);

	// Reading number of rows, columns and non-zero values in input matrix
	// users == rows
	// items == columns
	int users = parse_int(fp);
	int items = parse_int(fp);
	int non_zero = parse_int(fp);

	// Reading input matrix A
	mat2* A = mat2_new(users, items);
	for (int i = 0; i < non_zero; i++) {
		int row = parse_int(fp);
		int column = parse_int(fp);
		double value = parse_double(fp);
		mat2_set(A, row, column, value);
	}

	if (fclose(fp) == EOF) {
		die("Unable to close input file.\n");
	}

	// Creating L and R matrices and their auxiliaries
	mat2* L = mat2_new(users, features);
	mat2* R = mat2_new(features, items);
	mat2_random_fill(L, features);
	mat2_random_fill(R, features);

	mat2_print(A);
	mat2_print(L);
	mat2_print(R);

	mat2_free(A);
	mat2_free(L);
	mat2_free(R);

	// mat2* L_aux = mat2_new(users, features);
	// mat2_copy(L, L_aux);
	// mat2* R_aux = mat2_new(features, items);
	// mat2_copy(R, R_aux);

	return 0;
}
