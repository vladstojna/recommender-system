// serial implementation
#include "util.h"
#include "mat2d.h"
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

int col_cmp(const void *a, const void *b)
{
	non_zero_entry i_a = *(non_zero_entry *)a;
	non_zero_entry i_b = *(non_zero_entry *)b;

	if (i_a.col != i_b.col)
		return i_a.col - i_b.col;
	else
		return i_a.row - i_b.row;
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
	for (int i = 0; i < non_zero; i++)
	{
		int row = parse_int(fp);
		int column = parse_int(fp);
		double value = parse_double(fp);

		non_zero_entry in = {row, column, value};
		user_major[i] = in;
		item_major[i] = in;
		// entries[i] = in;
	}

	// Order item_major by items over users
	qsort((void *)item_major, non_zero, sizeof(non_zero_entry), col_cmp);

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
#define DP

#ifdef DP
	mat2d *dots = mat2d_new(users, items);
	bool *was_calc = (bool *)calloc(users * items, sizeof(bool));
#endif

	//__start_benchmark;
	for (int iter = 0; iter < iters; iter++)
	{

		int prev_i = -1;
		int prev_j = -1;

		for (int idx = 0; idx < non_zero; idx++)
		{
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

#ifndef DP
			double B_ij_L = mat2d_dot_product(L_aux, L_i, R_aux, L_j);
			double B_ij_R = (R_i == L_i && R_j == L_j) ? B_ij_L : mat2d_dot_product(L_aux, R_i, R_aux, R_j);
#endif

#ifdef DP
			double B_ij_L;
			double B_ij_R;

			if (ptr2d_get(was_calc, L_i, L_j, items))
			{
				B_ij_L = mat2d_get(dots, L_i, L_j);
			}
			else
			{
				B_ij_L = mat2d_dot_product(L_aux, L_i, R_aux, L_j);
				ptr2d_set(was_calc, L_i, L_j, items, true);
				mat2d_set(dots, L_i, L_j, B_ij_L);
			}

			if (R_i == L_i && R_j == L_j)
			{
				B_ij_R = B_ij_L;
			}
			else if (ptr2d_get(was_calc, R_i, R_j, items))
			{
				B_ij_R = mat2d_get(dots, R_i, R_j);
			}
			else
			{
				B_ij_R = mat2d_dot_product(L_aux, R_i, R_aux, R_j);
				ptr2d_set(was_calc, R_i, R_j, items, true);
				mat2d_set(dots, R_i, R_j, B_ij_R);
			}
#endif

			for (int k = 0; k < features; k++)
			{
				mat2d_set(L, L_i, k, mat2d_get(L, L_i, k) - alpha * 2 * (L_value - B_ij_L) * (-mat2d_get(R_aux, L_j, k)));

				mat2d_set(R, R_j, k, mat2d_get(R, R_j, k) - alpha * 2 * (R_value - B_ij_R) * (-mat2d_get(L_aux, R_i, k)));
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

#ifdef DP
		memset(was_calc, false, sizeof(bool) * users * items);
#endif
	}

	mat2d_prod(L, R, B);

	mat2d_print(L_aux);
	mat2d_print(R_aux);
	mat2d_print(B);

	//__end_benchmark("oof", 1);

	mat2d_free(L_aux);
	mat2d_free(R_aux);
	mat2d_free(B);
	mat2d_free(L);
	mat2d_free(R);
#ifdef DP
	mat2d_free(dots);
	free(was_calc);
#endif

	return 0;
}
