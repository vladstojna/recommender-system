// serial implementation
#include "util.h"
#include "mat2d.h"

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

	printf("iters: %d, alpha: %f, features: %d, users: %d, items: %d, non_zero: %d", iters, alpha, features, users, items, non_zero);

	// Reading input matrix A
	mat2d* A = mat2d_new(users, items);
	for (int i = 0; i < non_zero; i++) {
		int row = parse_int(fp);
		int column = parse_int(fp);
		double value = parse_double(fp);
		mat2d_set(A, row, column, value);
	}

	if (fclose(fp) == EOF) {
		die("Unable to close input file.\n");
	}

	// Creating L and R matrices and their auxiliaries
	mat2d* L = mat2d_new(users, features);
	mat2d* R = mat2d_new(features, items);
	mat2d_random_fill_LR(L, R, features);

	mat2d_print(A);
	mat2d_print(L);
	mat2d_print(R);

	mat2d* B = mat2d_new(users, items);
	mat2d_prod(L, R, B);
	mat2d_print(B);
	mat2d_free(B);

	mat2d_free(A);
	mat2d_free(L);
	mat2d_free(R);

	return 0;
}
