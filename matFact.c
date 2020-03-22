// serial implementation
#include "util.h"
#include "mat2d.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
	int row;
	int col;
	double value;
} info;

int col_cmp(const void * a, const void * b) {
	info i_a = *(info*)a;
	info i_b = *(info*)b;

	if (i_a.col != i_b.col)
		return i_a.col - i_b.col;
	else return i_a.row - i_b.row;
}

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

	info row_major[non_zero];
	info col_major[non_zero];

	// Reading input matrix A
	for (int i = 0; i < non_zero; i++) {
		int row = parse_int(fp);
		int column = parse_int(fp);
		double value = parse_double(fp);

		info in = { row, column, value };
		row_major[i] = in;
		col_major[i] = in;
	}

	qsort((void*)col_major, non_zero, sizeof(info), col_cmp);

	if (fclose(fp) == EOF) {
		die("Unable to close input file.\n");
	}

	// Creating L and R matrices and their auxiliaries
	mat2d* L = mat2d_new(users, features);
	mat2d* R_init = mat2d_new(features, items);
	mat2d_random_fill_LR(L, R_init, features);

	mat2d* R = mat2d_transpose(R_init);
	mat2d_free(R_init);

	mat2d* L_aux = mat2d_new(users, features);
	mat2d* R_aux = mat2d_new(items, features);
	mat2d_copy(L, L_aux);
	mat2d_copy(R, R_aux);

	mat2d* B = mat2d_new(users, items);
	mat2d_prod(L, R, B);

	for (int iter = 0; iter < iters; iter++) {

		for (int i = 0; i < non_zero;) {
			int curr_row = row_major[i].row;

			int skip;
			for (int k = 0; k < L->n_c; k++) {

				double sum = 0;
				skip = i;
				while (skip < non_zero && row_major[skip].row == curr_row) {
					sum += (row_major[skip].value - mat2d_get(B, row_major[skip].row, row_major[skip].col)) * (-mat2d_get(R_aux, row_major[skip].col, k));

					skip++;
				}

				mat2d_set(L, row_major[i].row, k, mat2d_get(L_aux, row_major[i].row, k) - alpha * 2 * sum);
			}
			i = skip;
		}

		for (int j = 0; j < non_zero;) {
			int curr_col = col_major[j].col;

			int skip;
			for (int k = 0; k < R->n_c; k++) {

				double sum = 0;
				skip = j;
				while (skip < non_zero && col_major[skip].col == curr_col) {
					sum += (col_major[skip].value - mat2d_get(B, col_major[skip].row, col_major[skip].	col)) * (-mat2d_get(L_aux, col_major[skip].row, k));

					skip++;
				}

				mat2d_set(R, col_major[j].col, k, mat2d_get(R_aux, col_major[j].col, k) - alpha * 2 * sum);
			}
			j = skip;
		}

		mat2d *tmp = L_aux;
		L_aux = L;
		L = tmp;

		tmp = R_aux;
		R_aux = R;
		R = tmp;

		mat2d_prod(L_aux, R_aux, B);
	}

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
