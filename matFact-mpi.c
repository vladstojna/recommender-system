// mpi implementation
#include "util.h"
#include "mat2d.h"
#include "benchmark.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) \
			(BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(index,p,n) \
			(((p)*((index)+1)-1)/(n))
			
typedef struct
{
	int row;
	int col;
	double value;
} non_zero_entry;

typedef struct
{
	// iters, features, users, items, non_zero
	int params[5];
	double alpha;
	non_zero_entry *entries;
} input_info;

input_info *read_input(const char *file_name) {
	FILE *fp = fopen(file_name, "r");
	input_info *info = 0;

	if (fp == NULL)
		die("Unable to open input file.");

	info = malloc(sizeof(input_info));

	// Reading number of iterations
	parse_int(fp, &info->params[0]);

	// Reading learning rate
	parse_double(fp, &info->alpha);

	// Reading number of features
	parse_int(fp, &info->params[1]);

	// Reading number of rows, columns and non-zero values in input matrix
	// users == rows
	// items == columns
	parse_three_ints(fp, &info->params[2], &info->params[3], &info->params[4]);

	info->entries = malloc(sizeof(non_zero_entry) * info->params[4]);

	for (int i = 0; i < info->params[4]; i++) {
		int row, column;
		double value;
		parse_non_zero_entry(fp, &row, &column, &value);

		non_zero_entry entry = { row, column, value };
		info->entries[i] = entry;
	}

	if (fclose(fp) == EOF)
	{
		die("Unable to close input file.");
	}

	return info;
}

void print_output(mat2d *L, mat2d *R, mat2d *B, non_zero_entry *entries) {
	int users = mat2d_rows(B);
	int items = mat2d_cols(B);

	mat2d_prod(L, R, B);

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

void matrix_factorization(int id, int p, double *L, double *R, input_info *info) {
	int iters = info->params[0];
	int features = info->params[1];
	int users = info->params[2];
	int items = info->params[3];
	int nz_size = info->params[4];
	double alpha = info->alpha;

	double *L_zero = malloc(sizeof(double) * users * features);
	double *R_zero = malloc(sizeof(double) * items * features);

	int low = BLOCK_LOW(id, p, nz_size);
	int high = BLOCK_HIGH(id, p, nz_size);

	for (int iter = 0; iter < iters; iter++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(L, users * features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(R, items * features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		if (!id) {
			memcpy(L_zero, L, users * features * sizeof(double));
			memcpy(R_zero, R, items * features * sizeof(double));
		} else {
			memset(L_zero, 0, users * features * sizeof(double));
			memset(R_zero, 0, items * features * sizeof(double));
		}

		for (int n = low; n <= high; n++)
		{
			int i = info->entries[n].row;
			int j = info->entries[n].col;

			/************** dot product **************/
			double *row = &L[i * features];
			double *col = &R[j * features];
			double dot = 0;


			for (int i = 0; i < features; ++i) {
				dot += row[i] * col[i];
			}
			/*****************************************/

			double value = alpha * 2 * (info->entries[n].value - dot);

			for (int k = 0; k < features; k++) {
				L_zero[i * features + k] -= value * (- R[j * features + k]); 
				R_zero[j * features + k] -= value * (- L[i * features + k]);
			}
		}

		// MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(L_zero, L, users * features, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(R_zero, R, items * features, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}

	free(L_zero);
	free(R_zero);
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "Run ./matFact.out file");
		die("Missing input file name.");
	}

	int nproc, id;
	input_info *info = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	// Creating MPI non_zero type
	MPI_Datatype non_zero_type;
	int lengths[3] = { 1, 1, 1};
	const MPI_Aint displacements[3] = { 0, sizeof(int), 2 * sizeof(int) };
	MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_DOUBLE };
	MPI_Type_create_struct(3, lengths, displacements, types, &non_zero_type);
	MPI_Type_commit(&non_zero_type);

	if (!id) {
		info = read_input(argv[1]);
		if (!info)
			die("Unable to initialize parameters.");
	} else {
		info = malloc(sizeof(input_info));
	}

	MPI_Bcast(info->params, 5, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&info->alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if (id) {
		info->entries = malloc(sizeof(non_zero_entry) * info->params[4]);
	}

	MPI_Bcast(info->entries, info->params[4], non_zero_type, 0, MPI_COMM_WORLD);

	// Creating L and R
	mat2d *L = mat2d_new(info->params[2], info->params[1]);
	// R is always assumed transposed
	mat2d *R = mat2d_new(info->params[3], info->params[1]);

	// initialize both matrices
	if (!id) {
		mat2d *R_init = mat2d_new(info->params[1], info->params[3]);
		mat2d_random_fill_LR(L, R_init, info->params[1]);
		mat2d_transpose(R_init, R);
		mat2d_free(R_init);
	}

	// MPI_Barrier(MPI_COMM_WORLD);
	matrix_factorization(id, nproc, L->data, R->data, info);

	// FIXME: Free info and entries

	// print output
	if (!id) {
		mat2d *B = mat2d_new(info->params[2], info->params[3]);
		print_output(L, R, B, info->entries);
		mat2d_free(B);
	}

	mat2d_free(L);
	mat2d_free(R);

	MPI_Finalize();

	return 0;
}
