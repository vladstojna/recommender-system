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
		mat->data = (double*) malloc(sizeof(double) * rows * columns);

		if (!mat->data) {
			free(mat);
			return 0;
		}
	}

	for (int i = 0; i < mat->n_r; i++)
		for (int j = 0; j < mat->n_c; j++)
			mat2d_set(mat, i, j, 0);

	return mat;
}

void mat2d_free(mat2d* mat) {
	free(mat->data);
	free(mat);
}

void mat2d_copy(mat2d* from, mat2d* to) {
	memcpy(to->data, from->data, sizeof(double) * to->n_r * to->n_c);
}

void mat2d_print(mat2d* mat) {
	printf("\n");

	for (int i = 0; i < mat->n_r; i++) {
		for (int j = 0; j < mat->n_c; j++)
			printf("%f  ", mat2d_get(mat, i, j));
		
		printf("\n");
	}
}

void mat2d_random_fill(mat2d* mat, double norm) {
	srandom(0);

	for (int i = 0; i < mat->n_r; i++)
		for (int j = 0; j < mat->n_c; j++)
			mat2d_set(mat, i, j, RAND01 / (double) norm);
}

void mat2d_prod(mat2d* left, mat2d* right, mat2d* dest) {
	if (left->n_c != right->n_r)
		die("The given matrices can't be multiplied with each other.\n");

	double res = 0;
	for (int i = 0; i < left->n_r; i++) {
		for (int j = 0; j < right->n_c; j++) {
			for (int k = 0; k < left->n_c; k++) {
				res = res + mat2d_get(left, i, k) * mat2d_get(right, k, j);
			}

			mat2d_set(dest, i, j, res);
			res = 0;
		}
	}
}