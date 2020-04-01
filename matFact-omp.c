// parallel implementation using OpenMP
#include "util.h"
#include "mat2d.h"
#include "adjlst.h"
#include "benchmark.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* 0 - atomic; 1 - reduce R; 2 - reduce L; 3 - reduce both */
#define METHOD 0
#define REDUCTION METHOD >=1 && METHOD <= 3
#define ATOMIC METHOD == 0
#define REDUCE_R METHOD == 1
#define REDUCE_L METHOD == 2
#define REDUCE_BOTH METHOD == 3

#define swap(T, a, b) { T tmp = a; a = b; b = tmp; }

typedef struct
{
	int row;
	int col;
	double value;
} non_zero_entry;

int col_cmp(const void *a, const void *b) {
	non_zero_entry *nz_a = (non_zero_entry*) a;
	non_zero_entry *nz_b = (non_zero_entry*) b;

	return nz_a->col == nz_b->col ? nz_a->row - nz_b->row : nz_a->col - nz_b->col;
}

int row_cmp(const void *a, const void *b) {
	non_zero_entry *nz_a = (non_zero_entry*) a;
	non_zero_entry *nz_b = (non_zero_entry*) b;

	return nz_a->row == nz_b->row ? nz_a->col - nz_b->col : nz_a->row - nz_b->row;
}

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

	#ifndef METHOD
	#define METHOD 0 // default is atomic
	#endif

	#if REDUCTION
	mat2d **reduction_array;
	#endif

	#pragma omp parallel
	{

	#if REDUCTION

	int num_threads = omp_get_num_threads();
	int tid = omp_get_thread_num();

	#pragma omp single
	{
		#if REDUCE_BOTH
		reduction_array = malloc(2 * num_threads * sizeof(mat2d*));
		#else
		reduction_array = malloc(num_threads * sizeof(mat2d*));
		#endif
	}

	#endif

	#if REDUCE_L || REDUCE_BOTH
	reduction_array[tid] = mat2d_new(users, features);
	#endif
	#if REDUCE_R
	reduction_array[tid] = mat2d_new(items, features);
	#endif
	#if REDUCE_BOTH
	reduction_array[tid + num_threads] = mat2d_new(items, features);
	#endif

	for (int iter = 0; iter < iters; iter++)
	{
		mat2d_copy_parallel(L, L_stable);
		mat2d_copy_parallel(R, R_stable);
		#pragma omp barrier

		#if REDUCE_L || REDUCE_BOTH
		mat2d *partial_L = reduction_array[tid];
		#endif
		#if REDUCE_R
		mat2d *partial_R = reduction_array[tid];
		#endif
		#if REDUCE_BOTH
		mat2d *partial_R = reduction_array[tid + num_threads];
		#endif

		#if REDUCE_L || REDUCE_BOTH
		mat2d_zero(partial_L);
		#endif
		#if REDUCE_R || REDUCE_BOTH
		mat2d_zero(partial_R);
		#endif

		#pragma omp for
		for (int n = 0; n < nz_size; n++)
		{
			int i = entries[n].row;
			int j = entries[n].col;
			double value = alpha * 2 * (entries[n].value - mat2d_dot_product(L_stable, i, R_stable, j));

			for (int k = 0; k < features; k++) {
				#if ATOMIC || REDUCE_R
				#pragma omp atomic
				mat2d_set(L, i, k, mat2d_get(L, i, k) - value *
					(-mat2d_get(R_stable, j, k)));
				#elif REDUCE_L || REDUCE_BOTH
				mat2d_set(partial_L, i, k, mat2d_get(partial_L, i, k) - value *
					(-mat2d_get(R_stable, j, k)));
				#endif

				#if ATOMIC || REDUCE_L
				#pragma omp atomic
				mat2d_set(R, j, k, mat2d_get(R, j, k) - value *
					(-mat2d_get(L_stable, i, k)));
				#elif REDUCE_R || REDUCE_BOTH
				mat2d_set(partial_R, j, k, mat2d_get(partial_R, j, k) - value *
					(-mat2d_get(L_stable, i, k)));
				#endif
			}
		}

		#if REDUCTION

		for (int t = 0; t < num_threads; t++) {
			#if REDUCE_L || REDUCE_BOTH
			mat2d_sum(L, reduction_array[t]);
			#endif
			#if REDUCE_R
			mat2d_sum(R, reduction_array[t]);
			#endif
			#if REDUCE_BOTH
			mat2d_sum(R, reduction_array[t + num_threads]);
			#endif
		}
		#pragma omp barrier

		#endif
	}

	mat2d_prod(L, R, B);

	#if REDUCTION
	free(reduction_array[tid]);
	#endif
	#if REDUCE_BOTH
	free(reduction_array[tid + num_threads]);
	#endif

	}

	#if REDUCTION
	free(reduction_array);
	#endif

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

	__init_benchmark;
	__start_benchmark

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

	// qsort(entries, non_zero, sizeof(non_zero_entry), col_cmp);

	__end_benchmark("input", 1)

	__start_benchmark;
	matrix_factorization(B, L, R, entries, non_zero, iters, alpha);
	__end_benchmark("main loop", 1);

	// qsort(entries, non_zero, sizeof(non_zero_entry), row_cmp);

	// print output
	__start_benchmark
	print_output(B, entries);
	__end_benchmark("output", 1)

	mat2d_free(B);
	mat2d_free(L);
	mat2d_free(R);
	free(entries);

	return 0;
}

