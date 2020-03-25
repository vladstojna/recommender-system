// serial implementation
#include "util.h"
#include "mat2d.h"
#include "adjlst.h"
#include "benchmark.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <memory.h>

//__init_benchmark;

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

	// non_zero_entry entries[non_zero];
	non_zero_entry *entries = (non_zero_entry *)malloc(sizeof(non_zero_entry) * non_zero);

	// Reading input matrix A
	adj_lst *A = adjlst_new(users);

	int prev_row = -1;
	int col_count = 0;
	// Reading input matrix A
	for (int i = 0; i < non_zero; i++, col_count++) {
		int row, column;
		double value;
		parse_non_zero_entry(fp, &row, &column, &value);

		if (i == 0)
			prev_row = row;

		if (row != prev_row) {
			adjlst_entries_set(A, prev_row, adjlst_new_entries(col_count), col_count);
			prev_row = row;
			col_count = 0;
		}

		non_zero_entry in = { row, column, value };
		entries[i] = in;
	}
	adjlst_entries_set(A, prev_row, adjlst_new_entries(col_count), col_count);

	for (int i = 0, j = 0; i < non_zero; i++) {
		adj_lst_entry *item = &A->columns[entries[i].row];
		column_entry *entry = &item->entries[j++ % item->size];
		entry->at = entries[i].col;
		entry->value = entries[i].value;
	}

	free(entries);

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

	mat2d *L_aux = mat2d_new(users, features);
	mat2d *R_aux = mat2d_new(items, features);
	mat2d_copy(L, L_aux);
	mat2d_copy(R, R_aux);

	mat2d *B = mat2d_new(users, items);

	bool *check_col = (bool *)calloc(items, sizeof(bool));

	__init_benchmark;
	__start_benchmark;

	for (int iter = 0; iter < iters; iter++)
	{
		for (size_t idx = 0; idx < A->rows; idx++)
		{
			if (adjlst_entries_sz(A, idx) == 0)
				continue;

			mat2d_set_line(L, idx, mat2d_get_line(L_aux, idx));

			for (size_t jdx = 0; jdx < adjlst_entries_sz(A, idx); jdx++) {
				int j = A->columns[idx].entries[jdx].at;

				if (check_col[j] == false) {
					check_col[j] = true;
					mat2d_set_line(R, j, mat2d_get_line(R_aux, j));
				}

				double dot = mat2d_dot_product(L_aux, idx, R_aux, j);
				double value = adjlst_entries(A, idx)[jdx].value;

				for (int k = 0; k < features; k++) {
					mat2d_set(L, idx, k, mat2d_get(L, idx, k) - alpha * 2 * (value - dot) * (-mat2d_get(R_aux, j, k)));
					mat2d_set(R, j, k, mat2d_get(R, j, k) - alpha * 2 * (value - dot) * (-mat2d_get(L_aux, idx, k)));
				}
			}
		}

		memset(check_col, false, sizeof(bool) * items);

		mat2d *tmp = L_aux;
		L_aux = L;
		L = tmp;

		tmp = R_aux;
		R_aux = R;
		R = tmp;
	}

	__end_benchmark("main loop", 1e3);

	free(check_col);

	adjlst_free(A);

	mat2d_prod(L_aux, R_aux, B);

	mat2d_print(L_aux);
	mat2d_print(R_aux);
	mat2d_print(B);

	mat2d_free(L_aux);
	mat2d_free(R_aux);
	mat2d_free(B);
	mat2d_free(L);
	mat2d_free(R);

	return 0;
}
