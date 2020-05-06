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

void print_dataset_info(int rank, dataset_info *info) {
	printf("-----\nRank=%d\nIters=%d\nFeatures=%d\nUsers=%d\nItems=%d\nNon-zero size=%d\nAlpha=%f\n-----\n",
		rank, info->iters, info->features, info->users, info->items, info->non_zero_sz, info->alpha);
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

int read_input_metadata(FILE *fp, dataset_info *info) {
	if (info == NULL)
		return -1;

	// Reading number of iterations
	parse_int(fp, &info->iters);

	// Reading learning rate
	parse_double(fp, &info->alpha);

	// Reading number of features
	parse_int(fp, &info->features);

	// Reading number of rows, columns and non-zero values in input matrix
	// users == rows
	// items == columns
	parse_three_ints(fp, &info->users, &info->items, &info->non_zero_sz);

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

int read_non_zero_entries(FILE *fp, non_zero_entry *buffer, int buffsz, int blk_high) {
	int entries_read = 0;
	int row = 0;
	int column;
	double value;

	long ftell_before;

	do {

		if (entries_read > buffsz) {
			return -1;
		}

		/* peek next line to check if there was a row change */
		ftell_before = ftell(fp);

		int elems_read = fscanf(fp, "%d %d %lf", &row, &column, &value);
		if (elems_read == EOF) {
			return entries_read;
		}
		if (elems_read != 3) {
			return -1;
		}

		/* rewind read line when row exceeds HIGH value */
		if (row > blk_high) {
			fseek(fp, ftell_before - ftell(fp), SEEK_CUR);
			return entries_read;
		}

		buffer[entries_read++] = (non_zero_entry) { row, column, value };

	} while (1);
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "Run ./matFact.out file");
		die("Missing input file name.");
	}

	FILE *fp = fopen(argv[1], "r");
	if (fp == NULL)
	{
		die("Could not open file.");
	}

	int nproc, id;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	printf("Init rank = %d\n", id);

	MPI_Status status;
	MPI_Datatype non_zero_type;
	MPI_Datatype dataset_info_type;
	create_non_zero_entry(&non_zero_type);
	create_dataset_info(&dataset_info_type);

	printf("rank=%d : init MPI types\n", id);

	/* the original, not partitioned dataset information */
	dataset_info orig;

	/* information after decomposition */
	input_info local;

	/**
	 * temporary buffer to store read non-zero entries
	 * must be the maximum size a decomposed block can take up
	 */
	non_zero_entry *tmpbuffer;
	int tmpsize;

	if (is_root(id)) {
		if (read_input_metadata(fp, &orig) != 0) {
			die("Unable to initialize parameters.");
		}
		print_dataset_info(id, &orig);

		tmpsize = BLOCK_SIZE(nproc - 1, nproc, orig.users) * orig.items;
		tmpbuffer = malloc(sizeof(non_zero_entry) * tmpsize);

		int entries_read = read_non_zero_entries(fp, tmpbuffer, tmpsize,
			BLOCK_HIGH(id, nproc, orig.users));
		if (entries_read < 0) {
			die("Unable to read non-zero entries");
		}
		local.dataset_info = (dataset_info) { orig.iters, orig.features,
			BLOCK_SIZE(id, nproc, orig.users), BLOCK_SIZE(id, nproc, orig.items),
			entries_read, orig.alpha };
		local.entries = malloc(sizeof(non_zero_entry) * entries_read);
		memcpy(local.entries, tmpbuffer, sizeof(non_zero_entry) * entries_read);

		int next_id = id + 1;
		while (next_id < nproc) {

			entries_read = read_non_zero_entries(fp, tmpbuffer, tmpsize,
				BLOCK_HIGH(next_id, nproc, orig.users));
			if (entries_read < 0) {
				die("Unable to read non-zero entries");
			}
			dataset_info tmpinfo = (dataset_info) { orig.iters, orig.features,
				BLOCK_SIZE(next_id, nproc, orig.users), BLOCK_SIZE(next_id, nproc, orig.items),
				entries_read, orig.alpha };
			MPI_Send(&tmpinfo, 1, dataset_info_type, next_id, 0, MPI_COMM_WORLD);
			MPI_Send(&tmpbuffer, entries_read, non_zero_type, next_id, 1, MPI_COMM_WORLD);

			next_id++;
		}
	}

	if (!is_root(id)) {
		MPI_Recv(&local.dataset_info, 1, dataset_info_type, 0, 0, MPI_COMM_WORLD, &status);
	}

	print_dataset_info(id, &local.dataset_info);

	if (!is_root(id)) {
		local.entries = malloc(sizeof(non_zero_entry) * local.dataset_info.non_zero_sz);
		MPI_Recv(local.entries, local.dataset_info.non_zero_sz, non_zero_type, 0, 1,
			MPI_COMM_WORLD, &status);
	}

	printf("rank = %d : received non-zero entries\n", id);

	/*

	// Creating L and R
	mat2d *L = mat2d_new(orig.dataset_info.users, orig.dataset_info.features);
	// R is always assumed transposed
	mat2d *R = mat2d_new(orig.dataset_info.items, orig.dataset_info.features);

	// initialize both matrices
	if (is_root(id)) {
		mat2d *R_init = mat2d_new(orig.dataset_info.features, orig.dataset_info.items);
		mat2d_random_fill_LR(L, R_init, orig.dataset_info.features);
		mat2d_transpose(R_init, R);
		mat2d_free(R_init);
	}

	printf("Initialized L and R, rank = %d\n", id);

	if (fclose(fp) == EOF)
		exit(1);

	// MPI_Barrier(MPI_COMM_WORLD);
	matrix_factorization(id, nproc, L->data, R->data, &orig);

	// print output
	if (is_root(id)) {
		mat2d *B = mat2d_new(orig.dataset_info.users, orig.dataset_info.items);
		mat2d_prod(L, R, B);
		print_output(B, orig.entries);
		mat2d_free(B);
	}

	free(orig.entries);
	mat2d_free(L);
	mat2d_free(R);

	*/

	MPI_Finalize();

	return 0;
}
