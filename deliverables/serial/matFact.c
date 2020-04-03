// serial implementation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/****************************************************
 * Utility functions
 ***************************************************/

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

/****************************************************
 * mat2d
 ***************************************************/ 

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct
{
	int n_r;
	int n_c;
	double *data;
} mat2d;

#define mat2d_rows(m) m->n_r
#define mat2d_cols(m) m->n_c
#define mat2d_get(m, r, c) m->data[((r)*m->n_c) + (c)]
#define mat2d_set(m, r, c, v) m->data[((r)*m->n_c) + (c)] = v

mat2d* mat2d_new(int rows, int columns) {
	mat2d *mat = malloc(sizeof(mat2d));
	
	if (mat) {
		mat->n_r = rows;
		mat->n_c = columns;
		mat->data = malloc(rows * columns * sizeof(double));

		if (!mat->data) {
			free(mat);
			return 0;
		}
	}

	return mat;
}

void mat2d_free(mat2d *mat) {
	free(mat->data);
	free(mat);
}

double *mat2d_get_line(mat2d *mat, int line) {
	return &(mat->data[line * mat->n_c]);
}

void mat2d_copy(mat2d *from, mat2d *to) {
	memcpy(to->data, from->data, sizeof(double) * to->n_r * to->n_c);
}

void mat2d_zero(mat2d *mat) {
	memset(mat->data, 0, mat->n_r * mat->n_c * sizeof(double));
}

void mat2d_random_fill_LR(mat2d *L, mat2d *R, double norm) {
	srandom(0);

	int i, j;
	for (i = 0; i < L->n_r; i++)
		for (j = 0; j < L->n_c; j++)
			mat2d_set(L, i, j, RAND01 / (double) norm);

	for (i = 0; i < R->n_r; i++)
		for (j = 0; j < R->n_c; j++)
			mat2d_set(R, i, j, RAND01 / (double) norm);
}

void mat2d_prod(mat2d *left, mat2d *right, mat2d *dest) {
	if (left->n_c != right->n_c)
		die("The given matrices can't be multiplied with each other.");

	for (int i = 0; i < left->n_r; i++) {
		for (int j = 0; j < right->n_r; j++) {
			mat2d_set(dest, i, j, 0);
			for (int k = 0; k < left->n_c; k++) {
				mat2d_set(dest, i, j, mat2d_get(dest, i, j) + mat2d_get(left, i, k) * mat2d_get(right, j, k));
			}
		}
	}
}

void mat2d_transpose(mat2d *orig, mat2d *transpose) {
	if (orig->n_r != transpose->n_c || orig->n_c != transpose->n_r)
		die("Can't transpose into desired matrix.");

	for (int i = 0; i < orig->n_r; i++) {
		for (int j = 0; j < orig->n_c; j++) {
			mat2d_set(transpose, j, i, mat2d_get(orig, i, j));
		}
	}
}

double mat2d_dot_product(mat2d *left, int r, mat2d *right, int c) {
	if (left->n_c != right->n_c) {
		die("Cannot calculate dot product: invalid sizes.");
	}

	double *row = mat2d_get_line(left, r);
	double *col = mat2d_get_line(right, c);
	double result = 0;

	for (int i = 0; i < left->n_c; ++i) {
		result += row[i] * col[i];
	}
	return result;
}

/****************************************************
 * Main functions
 ***************************************************/ 

typedef struct
{
	int row;
	int col;
	double value;
} non_zero_entry;

void print_output(mat2d *B, non_zero_entry *entries) {
	int users = mat2d_rows(B);
	int items = mat2d_cols(B);
	for (int i = 0, aix = 0; i < users; i++) {
		int max = -1;
		for (int j = 0; j < items; j++) {
			if (!(entries[aix].row == i && entries[aix].col == j)) {
				if (max == -1 || mat2d_get(B, i, j) > mat2d_get(B, i, max)) {
					max = j;
				}
			} else {
				aix++;
			}
		}
		if (max != -1)
			printf("%d\n", max);
	}
}

void matrix_factorization(mat2d *B, mat2d *L, mat2d *R, non_zero_entry *entries, int nz_size, int iters, double alpha) {
	int users = mat2d_rows(B);
	int items = mat2d_cols(B);
	int features = mat2d_cols(L);
	mat2d *L_stable = mat2d_new(users, features);
	mat2d *R_stable = mat2d_new(items, features);

	for (int iter = 0; iter < iters; iter++)
	{
		mat2d_copy(L, L_stable);
		mat2d_copy(R, R_stable);

		for (int n = 0; n < nz_size; n++)
		{
			int i = entries[n].row;
			int j = entries[n].col;
			double value = alpha * 2 * (entries[n].value - mat2d_dot_product(L_stable, i, R_stable, j));

			for (int k = 0; k < features; k++) {
				mat2d_set(L, i, k, mat2d_get(L, i, k) - value *
					(-mat2d_get(R_stable, j, k)));
				mat2d_set(R, j, k, mat2d_get(R, j, k) - value *
					(-mat2d_get(L_stable, i, k)));
			}
		}
	}
	mat2d_prod(L, R, B);

	mat2d_free(L_stable);
	mat2d_free(R_stable);
}

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

	non_zero_entry *entries = malloc(sizeof(non_zero_entry) * non_zero);

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

	mat2d *B = mat2d_new(users, items);

	matrix_factorization(B, L, R, entries, non_zero, iters, alpha);

	// print output
	print_output(B, entries);

	mat2d_free(B);
	mat2d_free(L);
	mat2d_free(R);
	free(entries);

	return 0;
}
