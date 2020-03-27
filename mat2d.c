#include "mat2d.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void mat2d_set_line(mat2d *mat, int line, double *values) {
	memcpy(&(mat->data[line * mat->n_c]), values, mat->n_c * sizeof(double));
}

void mat2d_copy(mat2d *from, mat2d *to) {
	memcpy(to->data, from->data, sizeof(double) * to->n_r * to->n_c);
}

void mat2d_print(mat2d *mat) {
	printf("\n");

	for (int i = 0; i < mat->n_r; i++) {
		for (int j = 0; j < mat->n_c; j++)
			printf("%f  ", mat2d_get(mat, i, j));
		
		printf("\n");
	}
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

void mat2d_zero(mat2d *mat) {
	memset(mat->data, 0, mat->n_r * mat->n_c * sizeof(double));
}

// Assumes R is transposed
void mat2d_prod(mat2d *left, mat2d *right, mat2d *dest) {
	if (left->n_c != right->n_c)
		die("The given matrices can't be multiplied with each other.");

	#pragma omp for
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

// Assumes right matrix is transposed
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

