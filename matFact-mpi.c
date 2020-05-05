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

#define is_root(id) ((id) == 0)

typedef struct
{
	int row;
	int col;
	double value;
} non_zero_entry;

typedef struct
{
	int iters;
	int features;
	int users;
	int items;
	int non_zero_sz;
	double alpha;
} dataset_info;

void print_dataset_info(dataset_info *info) {
	printf("-----\nIters=%d\nFeatures=%d\nUsers=%d\nItems=%d\nNon-zero size=%d\nAlpha=%f\n-----\n",
		info->iters, info->features, info->users, info->items, info->non_zero_sz, info->alpha);
}

int create_non_zero_entry(MPI_Datatype *type)
{
	MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE };
	MPI_Aint offsets[2] = { offsetof(non_zero_entry, row), offsetof(non_zero_entry, value) };
	int blocklen[2] = { 2, 1 };

	MPI_Type_create_struct(2, blocklen, offsets, types, type);
	return MPI_Type_commit(type);
}

int create_dataset_info(MPI_Datatype *type)
{
	MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE };
	MPI_Aint offsets[2] = { offsetof(dataset_info, iters), offsetof(dataset_info, alpha) };
	int blocklen[2] = { 5, 1 };

	MPI_Type_create_struct(2, blocklen, offsets, types, type);
	return MPI_Type_commit(type);
}

typedef struct
{
	dataset_info dataset_info;
	non_zero_entry *entries;
} input_info;

int read_input(const char *file_name, input_info *info) {
	FILE *fp = fopen(file_name, "r");
	
	if (info == NULL)
		return -1;

	if (fp == NULL)
		return -1;

	// Reading number of iterations
	parse_int(fp, &info->dataset_info.iters);

	// Reading learning rate
	parse_double(fp, &info->dataset_info.alpha);

	// Reading number of features
	parse_int(fp, &info->dataset_info.features);

	// Reading number of rows, columns and non-zero values in input matrix
	// users == rows
	// items == columns
	parse_three_ints(fp, &info->dataset_info.users,
		&info->dataset_info.items,
		&info->dataset_info.non_zero_sz);

	info->entries = malloc(sizeof(non_zero_entry) * info->dataset_info.non_zero_sz);

	for (int i = 0; i < info->dataset_info.non_zero_sz; i++) {
		int row, column;
		double value;
		parse_non_zero_entry(fp, &row, &column, &value);

		non_zero_entry entry = { row, column, value };
		info->entries[i] = entry;
	}

	if (fclose(fp) == EOF)
		return -1;

	return 0;
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

void matrix_factorization(int id, int p, double *L, double *R, input_info *info) {
	int iters = info->dataset_info.iters;
	int features = info->dataset_info.features;
	int users = info->dataset_info.users;
	int items = info->dataset_info.items;
	int nz_size = info->dataset_info.non_zero_sz;
	double alpha = info->dataset_info.alpha;

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

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	printf("Init rank = %d\n", id);

	input_info local;
	if (is_root(id)) {
		if (read_input(argv[1], &local) != 0) {
			die("Unable to initialize parameters.");
		}
		print_dataset_info(&local.dataset_info);
	}

	// Creating MPI types
	MPI_Datatype non_zero_type;
	MPI_Datatype dataset_info_type;
	create_non_zero_entry(&non_zero_type);
	create_dataset_info(&dataset_info_type);

	MPI_Bcast(&local.dataset_info, 1, dataset_info_type, 0, MPI_COMM_WORLD);

	printf("Received dataset info, rank = %d\n", id);

	if (!is_root(id)) {
		local.entries = malloc(sizeof(non_zero_entry) * local.dataset_info.non_zero_sz);
	}

	MPI_Bcast(local.entries, local.dataset_info.non_zero_sz, non_zero_type, 0, MPI_COMM_WORLD);

	printf("Received non-zero entries, rank = %d\n", id);

	// Creating L and R
	mat2d *L = mat2d_new(local.dataset_info.users, local.dataset_info.features);
	// R is always assumed transposed
	mat2d *R = mat2d_new(local.dataset_info.items, local.dataset_info.features);

	// initialize both matrices
	if (is_root(id)) {
		mat2d *R_init = mat2d_new(local.dataset_info.features, local.dataset_info.items);
		mat2d_random_fill_LR(L, R_init, local.dataset_info.features);
		mat2d_transpose(R_init, R);
		mat2d_free(R_init);
	}

	printf("Initialized L and R, rank = %d\n", id);

	// MPI_Barrier(MPI_COMM_WORLD);
	matrix_factorization(id, nproc, L->data, R->data, &local);

	// print output
	if (is_root(id)) {
		mat2d *B = mat2d_new(local.dataset_info.users, local.dataset_info.items);
		mat2d_prod(L, R, B);
		print_output(B, local.entries);
		mat2d_free(B);
	}

	free(local.entries);
	mat2d_free(L);
	mat2d_free(R);

	MPI_Finalize();

	return 0;
}
