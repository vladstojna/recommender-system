// serial implementation
#include "util.h"
#include "mat2d.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
	int row;
	int col;
	double value;
} non_zero_entry;

int col_cmp(const void * a, const void * b) {
	non_zero_entry i_a = *(non_zero_entry*)a;
	non_zero_entry i_b = *(non_zero_entry*)b;

	if (i_a.col != i_b.col)
		return i_a.col - i_b.col;
	else return i_a.row - i_b.row;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "Run ./matFact.out file");
		die("Missing input file name.");
	}

	const char *file_name = argv[1];
	FILE *fp = fopen(file_name, "r");

	if (fp == NULL)
		die("Unable to open input file.");

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

	// non_zero_entry entries[non_zero];
	non_zero_entry user_major[non_zero];
	non_zero_entry item_major[non_zero];

	// Reading input matrix A
	for (int i = 0; i < non_zero; i++) {
		int row = parse_int(fp);
		int column = parse_int(fp);
		double value = parse_double(fp);

		non_zero_entry in = { row, column, value };
		user_major[i] = in;
		item_major[i] = in;
		// entries[i] = in;
	}

	// Order item_major by items over users
	qsort((void*)item_major, non_zero, sizeof(non_zero_entry), col_cmp);

	if (fclose(fp) == EOF) {
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
	mat2d_prod(L, R, B);

	for (int iter = 0; iter < iters; iter++) {

		int prev_i = -1;
		int prev_j = -1;

		for (int idx = 0; idx < non_zero; idx++) {
			int L_i = user_major[idx].row;
			int L_j = user_major[idx].col;

			int R_i = item_major[idx].row;
			int R_j = item_major[idx].col;

			if (L_i != prev_i)
				mat2d_set_line(L, L_i, mat2d_get_line(L_aux, L_i));

			if (R_j != prev_j)
				mat2d_set_line(R, R_j, mat2d_get_line(R_aux, R_j));

			double L_value = user_major[idx].value;
			double R_value = item_major[idx].value;

			for (int k = 0; k < features; k++) {
				mat2d_set(L, L_i, k, mat2d_get(L, L_i, k) - alpha * 2 * 
				(L_value - mat2d_get(B, L_i, L_j)) * (-mat2d_get(R_aux, L_j, k)));

				mat2d_set(R, R_j, k, mat2d_get(R, R_j, k) - alpha * 2 * 
				(R_value - mat2d_get(B, R_i, R_j)) * (-mat2d_get(L_aux, R_i, k)));
			}

			prev_i = L_i;
			prev_j = R_j;
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
