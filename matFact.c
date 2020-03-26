// serial implementation
#include "util.h"
#include "mat2d.h"
#include "adjlst.h"
#include "benchmark.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	int row;
	int col;
	double value;
} non_zero_entry;

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "Run ./matFact.out file");
		die("Missing input file name.");
	}

	const char *file_name = argv[1];
	FILE *fp = fopen(file_name, "r");

	if (fp == NULL)
		die("Unable to open input file.");

	// Reading number of iterations
	int iters;
	parse_int(fp, &iters);

	// Reading learning rate
	double alpha;
	parse_double(fp, &alpha);

	// Reading number of features
	int features;
	parse_int(fp, &features);

	// Reading number of rows, columns and non-zero values in input matrix
	// users == rows
	// items == columns
	int users, items, non_zero;
	parse_three_ints(fp, &users, &items, &non_zero);

	non_zero_entry entries[non_zero];

	for (int i = 0; i < non_zero; i++) {
		int row, column;
		double value;
		parse_non_zero_entry(fp, &row, &column, &value);

		non_zero_entry entry = { row, column, value };
		entries[i] = entry;
	}

	if (fclose(fp) == EOF)
	{
		die("Unable to close input file.");
	}

	// Creating L and R matrices and their auxiliaries
	mat2d *L = mat2d_new(users, features);
	mat2d *R_init = mat2d_new(features, items);
	mat2d_random_fill_LR(L, R_init, features);

	// R is always assumed transposed
	mat2d *R = mat2d_new(items, features);
	mat2d_transpose(R_init, R);
	mat2d_free(R_init);

	mat2d *L_stable = mat2d_new(users, features);
	mat2d *R_stable = mat2d_new(items, features);
	mat2d_copy(L, L_stable);
	mat2d_copy(R, R_stable);

	mat2d *B = mat2d_new(users, items);

	__init_benchmark;
	__start_benchmark;

	for (int iter = 0; iter < iters; iter++)
	{
		mat2d_zero(L);
		mat2d_zero(R);

		for (int n = 0; n < non_zero; n++)
		{
			int i = entries[n].row;
			int j = entries[n].col;
			double dot = mat2d_dot_product(L_stable, i, R_stable, j);
			double value = entries[n].value;

			for (int k = 0; k < features; k++) {
				mat2d_set(L, i, k, mat2d_get(L, i, k) + (value - dot) * 
				(-mat2d_get(R_stable, j, k)));
				mat2d_set(R, j, k, mat2d_get(R, j, k) + (value - dot) * 
				(-mat2d_get(L_stable, i, k)));
			}
		}

		for (int r = 0; r < users; r++)
			for (int c = 0; c < features; c++)
				mat2d_set(L, r, c, mat2d_get(L_stable, r, c) - alpha * 2 * mat2d_get(L, r, c));

		for (int r = 0; r < items; r++)
			for (int c = 0; c < features; c++)
				mat2d_set(R, r, c, mat2d_get(R_stable, r, c) - alpha * 2 * mat2d_get(R, r, c));

		mat2d *tmp = L_stable;
		L_stable = L;
		L = tmp;

		tmp = R_stable;
		R_stable = R;
		R = tmp;
	}

	__end_benchmark("main loop", 1e3);

	mat2d_prod(L_stable, R_stable, B);

	// mat2d_print(L_stable);
	// mat2d_print(R_stable);
	// mat2d_print(B);

	mat2d_free(L_stable);
	mat2d_free(R_stable);
	mat2d_free(B);
	mat2d_free(L);
	mat2d_free(R);

	return 0;
}
