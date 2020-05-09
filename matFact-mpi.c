// mpi implementation
#include "util.h"
#include "mat2d.h"
#include "benchmark.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) \
			(BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(index,p,n) \
			(((p)*((index)+1)-1)/(n))

#define is_root(id) ((id) == 0)

typedef struct
{
	int index;
	double value;
} output_entry;

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
	int users_init;
	int items_init;
	int non_zero_sz;
	double alpha;
} dataset_info;

typedef struct
{
	int rows;
	int cols;
	int rows_periodic;
	int cols_periodic;
	int row;
	int col;
} grid_info;

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

void max_cmp(output_entry *in, output_entry *inout, int *len, MPI_Datatype *type) {
	int sz = *len;
	for (int i = 0; i < sz; i++) {
		inout[i] = (in[i].value > inout[i].value ? in[i] : inout[i]);
	}
}

void print_dataset_info(int rank, dataset_info *info) {
	printf("----- Rank=%d Iters=%d Features=%d Users=%d(%d) Items=%d(%d) Non-zero size=%d Alpha=%f -----\n",
		rank, info->iters, info->features,
		info->users, info->users_init, info->items,
		info->items_init, info->non_zero_sz, info->alpha);
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
	int blocklen[2] = { 7, 1 };

	MPI_Type_create_struct(2, blocklen, offsets, types, type);
	return MPI_Type_commit(type);
}

int create_output_entry(MPI_Datatype *type)
{
	MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE };
	MPI_Aint offsets[2] = { offsetof(output_entry, index), offsetof(output_entry, value) };
	int blocklen[2] = { 1, 1 };
	MPI_Type_create_struct(2, blocklen, offsets, types, type);
	return MPI_Type_commit(type);
}

void create_cart_comm(MPI_Comm *comm, int nproc)
{
	int size[] = { 0, 0 };
	int periodic[] = { 0, 0 };
	MPI_Dims_create(nproc, 2, size);
	MPI_Cart_create(MPI_COMM_WORLD, 2, size, periodic, 1, comm);
}

void split_comms(MPI_Comm cart_comm, MPI_Comm *row_comm, MPI_Comm *col_comm, int rank)
{
	int coords[2];
	MPI_Cart_coords(cart_comm, rank, 2, coords);
	MPI_Comm_split(cart_comm, coords[0], coords[1], row_comm);
	MPI_Comm_split(cart_comm, coords[1], coords[0], col_comm);
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

void compute_output(
		MPI_Comm cart_comm,
		MPI_Comm row_comm,
		MPI_Comm col_comm,
		MPI_Datatype output_entry_type,
		MPI_Op reduction_op,
		const dataset_info *info,
		const non_zero_entry *entries,
		const grid_info *grid,
		mat2d *L,
		mat2d *R)
{
	output_entry *out;
	output_entry *red_res;

	int rank;
	int row_rank;
	int col_rank;
	MPI_Comm_rank(cart_comm, &rank);
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_rank(col_comm, &col_rank);

	int usersz = BLOCK_SIZE(grid->row, grid->rows, info->users_init);
	int itemsz = BLOCK_SIZE(grid->col, grid->cols, info->items_init);

	int offset_row = BLOCK_LOW(grid->row, grid->rows, info->users_init);
	int offset_col = BLOCK_LOW(grid->col, grid->cols, info->items_init);

	if (is_root(row_rank)) {
		red_res = malloc(sizeof(output_entry) * usersz);
	}

	/* the array with the partial output */
	out = malloc(sizeof(output_entry) * usersz);
	/* initialize the max array with a marker */
	for (int i = 0; i < usersz; i++) {
		out[i] = (output_entry) { -1, -1 };
	}

	for (int i = 0, aix = 0; i < usersz; i++) {
		output_entry max = { -1, -1 };
		for (int j = 0; j < itemsz; j++) {
			if (!(entries[aix].row == offset_row + i && entries[aix].col == offset_col + j)) {
				double dot = mat2d_dot_product(L, i, R, j);
				if (dot > max.value) {
					max.value = dot;
					max.index = offset_col + j;
				}
			} else {
				aix++;
			}
		}
		out[i] = max;
	}

	MPI_Reduce(out, red_res, usersz, output_entry_type, reduction_op, 0, row_comm);

	free(out);

	output_entry *gathered_output = NULL;
	int *recvcounts = NULL;
	int *displs = NULL;

	if (is_root(rank)) {
		/* the array where the partial outputs will be gathered in */
		gathered_output = malloc(sizeof(output_entry) * info->users_init);
		recvcounts = malloc(sizeof(int) * grid->rows);
		displs = malloc(sizeof(int) * grid->rows);
		
		for (int i = 0; i < grid->rows; i++) {
			displs[i] = BLOCK_LOW(i, grid->rows, info->users_init);
			recvcounts[i] = BLOCK_SIZE(i, grid->rows, info->users_init);
		}
		memcpy(gathered_output, red_res, sizeof(output_entry) * usersz);
	}
	
	if (rank % grid->cols == 0) {
		MPI_Gatherv(red_res, usersz, output_entry_type, gathered_output, recvcounts, displs, output_entry_type, 0, col_comm);
	}

	if (is_root(row_rank)) {
		free(red_res);
	}

	if (is_root(rank)) {
		for (int i = 0; i < info->users_init; i++) {
			output_entry max = gathered_output[i];
			if (max.index != -1) {
				printf("%d\n", max.index);
			}
		}
	}

	if (is_root(rank)) {
		free(gathered_output);
	}
}

void matrix_factorization(
		MPI_Comm row_comm,
		MPI_Comm col_comm,
		mat2d *L,
		mat2d *R,
		const dataset_info *info,
		const non_zero_entry *entries,
		const grid_info *grid)
{
	int iters = info->iters;
	int features = info->features;
	int users = info->users;
	int items = info->items;
	int nz_size = info->non_zero_sz;
	double alpha = info->alpha;

	int offset_row = BLOCK_LOW(grid->row, grid->rows, info->users_init);
	int offset_col = BLOCK_LOW(grid->col, grid->cols, info->items_init);

	mat2d *L_aux = mat2d_new(users, features);
	mat2d *R_aux = mat2d_new(items, features);

	int row_rank;
	int col_rank;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_rank(col_comm, &col_rank);

	for (int iter = 0; iter < iters; iter++)
	{
		MPI_Barrier(MPI_COMM_WORLD);

		is_root(col_rank) ? mat2d_copy(R, R_aux) : mat2d_zero(R_aux);
		is_root(row_rank) ? mat2d_copy(L, L_aux) : mat2d_zero(L_aux);

		for (int n = 0; n < nz_size; n++)
		{
			int i = entries[n].row - offset_row;
			int j = entries[n].col - offset_col;
			double value = alpha * 2 * (entries[n].value - mat2d_dot_product(L, i, R, j));

			for (int k = 0; k < features; k++) {

				mat2d_set(L_aux, i, k, mat2d_get(L_aux, i, k) - value *
					(-mat2d_get(R, j, k)));
				mat2d_set(R_aux, j, k, mat2d_get(R_aux, j, k) - value *
					(-mat2d_get(L, i, k)));
			}
		}

		MPI_Allreduce(mat2d_data(L_aux), mat2d_data(L), mat2d_size(L), MPI_DOUBLE, MPI_SUM, row_comm);
		MPI_Allreduce(mat2d_data(R_aux), mat2d_data(R), mat2d_size(R), MPI_DOUBLE, MPI_SUM, col_comm);
	}

	mat2d_free(L_aux);
	mat2d_free(R_aux);
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

void distribute_non_zero_values(
		FILE *fp,
		MPI_Comm cart_comm,
		MPI_Datatype non_zero_type,
		MPI_Datatype dataset_info_type,
		int rows,
		int cols,
		input_info *local,
		const dataset_info *orig)
{
	/**
	 * temporary buffer to store read non-zero entries
	 * size must be the maximum number of non-zero elements between the lines of a decomposed block
	 */
	int tmpsize = BLOCK_SIZE(rows - 1, rows, orig->users) * orig->items;
	non_zero_entry *tmpbuffer = malloc(sizeof(non_zero_entry) * tmpsize);

	/**
	 * if no non-zero entries in grid block, send something back to
	 * avoid the receiving process staying blocked forever
	 * save in this array which ranks have been sent to
	 */
	int nprocs;
	MPI_Comm_size(cart_comm, &nprocs);
	char is_sent[nprocs];
	memset(is_sent, 0, nprocs);

	for (int row = 0; row < rows; row++) {

		int entries_read = read_non_zero_entries(fp, tmpbuffer, tmpsize,
			BLOCK_HIGH(row, rows, orig->users));
		if (entries_read < 0) {
			die("Unable to read non-zero entries");
		}

		/* sort by column to find the frontier of each grid element */
		qsort(tmpbuffer, entries_read, sizeof(non_zero_entry), col_cmp);

		non_zero_entry *base = tmpbuffer;
		for (int i = 0, col = 0; i < entries_read; i++) {

			if (col > cols) {
				die("More columns read than grid width");
			}

			non_zero_entry *frontier;
			if (i < entries_read - 1)
				frontier = &tmpbuffer[i + 1];
			else
				frontier = &tmpbuffer[i];

			int rank;
			int coords[] = { row, col };
			MPI_Cart_rank(cart_comm, coords, &rank);

			if (i == entries_read - 1 || frontier->col > BLOCK_HIGH(col, cols, orig->items)) {

				/**
				 * if it is the last entry in the buffer
				 * need to consider one more value to have correct size
				 */
				size_t offset = frontier - base + (i == entries_read - 1);

				/* sort by line before storing or sending */
				qsort(base, offset, sizeof(non_zero_entry), row_cmp);

				int blk_sz_users = BLOCK_SIZE(row, rows, orig->users);
				int blk_sz_items = BLOCK_SIZE(col, cols, orig->items);

				dataset_info tmpinfo = (dataset_info) {
					orig->iters,
					orig->features,
					blk_sz_users,
					blk_sz_items,
					orig->users,
					orig->items,
					offset,
					orig->alpha };

				/* root simply copies the contents to its local structures */
				if (is_root(rank)) {
					local->dataset_info = tmpinfo;
					local->entries = malloc(sizeof(non_zero_entry) * offset);
					memcpy(local->entries, base, sizeof(non_zero_entry) * offset);
				} else {
					MPI_Send(&tmpinfo, 1, dataset_info_type, rank, 0, MPI_COMM_WORLD);
					MPI_Send(base, offset, non_zero_type, rank, 1, MPI_COMM_WORLD);
				}

				is_sent[rank] = 1;
				base = frontier;
				col++;
			}
		}
	}
	free(tmpbuffer);

	/* send some data to ranks which hadn't been sent to yet */
	dataset_info ds_info_to_send = {
		.iters = orig->iters,
		.features = orig->features,
		.users = 0,
		.items = 0,
		.users_init = orig->users,
		.items_init = orig->items,
		.non_zero_sz = 0,
		.alpha = orig->alpha };

	int coords[2];
	for (int r = 0; r < nprocs; r++) {
		if (is_sent[r] == 0) {
			MPI_Cart_coords(cart_comm, r, 2, coords);
			ds_info_to_send.users = BLOCK_SIZE(coords[0] /*row*/, rows, orig->users);
			ds_info_to_send.items = BLOCK_SIZE(coords[1] /*col*/, cols, orig->items);
			MPI_Send(&ds_info_to_send, 1, dataset_info_type, r, 0, MPI_COMM_WORLD);
		}
	}
}

void receive_non_zero_values(
		input_info *local,
		MPI_Datatype ds_info_type,
		MPI_Datatype nz_type,
		MPI_Status *status)
{
	MPI_Recv(&local->dataset_info, 1, ds_info_type, 0, 0, MPI_COMM_WORLD, status);
	if (local->dataset_info.non_zero_sz > 0) {
		local->entries = malloc(sizeof(non_zero_entry) * local->dataset_info.non_zero_sz);
		MPI_Recv(local->entries, local->dataset_info.non_zero_sz, nz_type, 0, 1, MPI_COMM_WORLD, status);
	} else {
		local->entries = NULL;
	}
}

void distribute_matrix_L(MPI_Comm cart_comm, int rows, int users, int features) {
	double *buffer = malloc(sizeof(double) * BLOCK_SIZE(rows - 1, rows, users) * features);

	for (int row = 0; row < rows; row++) {
		int dest_rank;
		int coords[] = { row, 0 };
		MPI_Cart_rank(cart_comm, coords, &dest_rank);

		if (is_root(dest_rank))
			continue;

		int buffersz = BLOCK_SIZE(row, rows, users) * features;

		for (int j = 0; j < buffersz; j++) {
			buffer[j] = RAND01 / features;
		}

		MPI_Send(buffer, buffersz, MPI_DOUBLE, dest_rank, 2, cart_comm);
	}

	free(buffer);
}

void init_distribute_matrix_R(MPI_Comm cart_comm, mat2d *R_init, int cols, int items, int features) {
	double *buffer = malloc(sizeof(double) * items);

	for (int f = 0; f < features; f++) {

		for (int i = 0; i < items; i++) {
			buffer[i] = RAND01 / features;
		}

		for (int col = 0; col < cols; col++) {

			int dest_rank;
			int coords[] = { 0, col };
			MPI_Cart_rank(cart_comm, coords, &dest_rank);

			int blocksz = BLOCK_SIZE(col, cols, items);
			double *src_addr = buffer + BLOCK_LOW(col, cols, items);

			if (is_root(dest_rank)) {
				memcpy(mat2d_data(R_init)+ f * blocksz, src_addr, sizeof(double) * blocksz);
			} else {
				MPI_Send(src_addr, blocksz, MPI_DOUBLE, dest_rank, 3, cart_comm);
			}
		}
	}
	free(buffer);
}

void receive_matrix_R(MPI_Comm cart_comm, mat2d *R_init, int features, MPI_Status *status) {
	double *base = mat2d_data(R_init);
	for (int f = 0; f < features; f++, base += mat2d_cols(R_init)) {
		MPI_Recv(base, mat2d_cols(R_init), MPI_DOUBLE, 0, 3, cart_comm, status);
	}
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

	int nproc, rank, row_rank, col_rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	MPI_Comm cart_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;

	MPI_Status status;

	MPI_Datatype non_zero_type;
	MPI_Datatype dataset_info_type;

	create_non_zero_entry(&non_zero_type);
	create_dataset_info(&dataset_info_type);

	create_cart_comm(&cart_comm, nproc);
	MPI_Comm_rank(cart_comm, &rank);

	split_comms(cart_comm, &row_comm, &col_comm, rank);
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_rank(col_comm, &col_rank);

	grid_info grid;
	MPI_Cart_get(cart_comm, 2, &grid.rows, &grid.rows_periodic, &grid.row);

	printf("rank=%-3d : row-rank=%d col-rank=%d : dims %d x %d : coords (%d, %d)\n",
		rank, row_rank, col_rank, grid.rows, grid.cols, grid.row, grid.col);

	/* the original, not partitioned dataset information */
	dataset_info orig;

	/* information after decomposition */
	input_info local;

	/**
	 * L and R matrices
	 * R is always assumed transposed
	 */
	mat2d *L;
	mat2d *R;

	/**
	 * matrix initialization must follow a well defined order
	 * in order to converge correctly with a fixed number of iterations
	 * first, initialize L, by grid line order
	 * then initialize R, all columns per feature
	 */
	mat2d_init_seed();

	MPI_Barrier(cart_comm);

	if (is_root(rank)) {
		if (read_input_metadata(fp, &orig) != 0) {
			die("Unable to initialize parameters.");
		}
		orig.users_init = orig.users;
		orig.items_init = orig.items;
		print_dataset_info(rank, &orig);

		distribute_non_zero_values(fp,
			cart_comm,
			non_zero_type,
			dataset_info_type,
			grid.rows,
			grid.cols,
			&local,
			&orig);

		if (fclose(fp) == EOF)
			fprintf(stderr, "Unable to close file");
	} else {
		receive_non_zero_values(&local,
			dataset_info_type,
			non_zero_type,
			&status);
	}

	print_dataset_info(rank, &local.dataset_info);
	printf("rank=%-3d : received non-zero entries\n", rank);

	L = mat2d_new(local.dataset_info.users, local.dataset_info.features);

	if (is_root(rank)) {
		mat2d_random_fill(L, local.dataset_info.features);
		if (grid.rows > 1)
			distribute_matrix_L(cart_comm, grid.rows, orig.users, orig.features);
	} else if (grid.rows > 1 && rank % grid.cols == 0) {
		MPI_Recv(mat2d_data(L), mat2d_size(L), MPI_DOUBLE, 0, 2, cart_comm, &status);
	}

	MPI_Bcast(mat2d_data(L), mat2d_size(L), MPI_DOUBLE, 0, row_comm);
	printf("rank=%-3d : received matrix L block size=%d\n", rank, mat2d_size(L));

	mat2d *R_init = mat2d_new(local.dataset_info.features, local.dataset_info.items);

	if (is_root(rank)) {
		init_distribute_matrix_R(cart_comm, R_init, grid.cols, orig.items, orig.features);
	} else if (rank < grid.cols) {
		receive_matrix_R(cart_comm, R_init, local.dataset_info.features, &status);
	}

	MPI_Bcast(mat2d_data(R_init), mat2d_size(R_init), MPI_DOUBLE, 0, col_comm);

	R = mat2d_new(local.dataset_info.items, local.dataset_info.features);
	mat2d_transpose(R_init, R);
	mat2d_free(R_init);

	printf("rank=%-3d : received matrix R block size=%d\n", rank, mat2d_size(R));

	matrix_factorization(row_comm, col_comm, L, R, &local.dataset_info, local.entries, &grid);

	MPI_Datatype output_entry_type;
	create_output_entry(&output_entry_type);

	MPI_Op reduce_output_op;
	MPI_Op_create((MPI_User_function *) max_cmp, 1, &reduce_output_op);

	compute_output(cart_comm, row_comm, col_comm, output_entry_type, reduce_output_op, &local.dataset_info, local.entries, &grid, L, R);

	free(local.entries);
	mat2d_free(L);
	mat2d_free(R);

	MPI_Finalize();

	return 0;
}
