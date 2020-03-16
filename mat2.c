#include "mat2.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

mat2* mat2_new(int rows, int columns) {
	mat2 *mat = malloc(sizeof(mat2));
	
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
			mat2_set(mat, i, j, 0);

	return mat;
}

void mat2_free(mat2* mat) {
	free(mat->data);
	free(mat);
}

void mat2_copy(mat2* from, mat2* to) {
	memcpy(from->data, to->data, sizeof(double) * to->n_r * to->n_c);
}

void mat2_print(mat2* mat) {
	printf("\n");

	for (int i = 0; i < mat->n_r; i++) {
		for (int j = 0; j < mat->n_c; j++)
			printf("%f  ", mat2_get(mat, i, j));
		
		printf("\n");
	}
}

void mat2_random_fill(mat2* mat, double norm) {
	srandom(0);

	for (int i = 0; i < mat->n_r; i++)
		for (int j = 0; j < mat->n_c; j++)
			mat2_set(mat, i, j, RAND01 / (double) norm);
}

void mat2_prod(mat2* left, mat2* right, mat2* dest) {
	if (left->n_c != right->n_r)
		die("The given matrices can't be multiplied with each other.\n");

	double res = 0;
	for (int i = 0; i < left->n_r; i++) {
		for (int j = 0; j < right->n_c; j++) {
			for (int k = 0; k < left->n_c; k++) {
				res = res + mat2_get(left, i, k) * mat2_get(right, k, j);
			}

			mat2_set(dest, i, j, res);
			res = 0;
		}
	}
}