// hybrid implementation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <unistd.h>

/**********************************************************
 * util
 *********************************************************/
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

void memcpy_parallel(void *dest, const void *src, size_t sz, size_t type_sz, int tid, int num_threads) {
	size_t stride = sz / num_threads;
	memcpy((char*) dest + tid * stride * type_sz,
		(char*) src + tid * stride * type_sz,
		type_sz * ((tid + 1) == num_threads ? sz - stride * tid : stride));
}

void memset_parallel(void *buffer, int value, size_t sz, size_t type_sz, int tid, int num_threads) {
	size_t stride = sz / num_threads;
	memset((char*) buffer + tid * stride * type_sz,
		value,
		type_sz * ((tid + 1) == num_threads ? sz - stride * tid : stride));
}


/**********************************************************
 * mat2d
 *********************************************************/
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
#define mat2d_get_index(m, i) m->data[i]
#define mat2d_set_index(m, i, v) m->data[i] = v

#define mat2d_data(m) m->data
#define mat2d_size(m) (m->n_r * m->n_c)

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

void mat2d_copy_parallel(mat2d *from, mat2d *to, int tid, int num_threads) {
	memcpy_parallel(to->data, from->data, to->n_r * to->n_c, sizeof(double), tid, num_threads);
}

void mat2d_zero(mat2d *mat) {
	memset(mat->data, 0, mat->n_r * mat->n_c * sizeof(double));
}

void mat2d_zero_parallel(mat2d *mat, int tid, int num_threads) {
	memset_parallel(mat->data, 0, mat->n_r * mat->n_c, sizeof(double), tid, num_threads);
}

void mat2d_init_seed() {
	srandom(0);
}

void mat2d_random_fill(mat2d *m, double norm) {
	int i, j;
	for (i = 0; i < m->n_r; i++)
		for (j = 0; j < m->n_c; j++)
			mat2d_set(m, i, j, RAND01 / (double) norm);
}

void mat2d_sum(mat2d *res, mat2d *m) {
	if (mat2d_rows(res) != mat2d_rows(m) || mat2d_cols(res) != mat2d_cols(res))
		die("Cannot sum matrices.");

	int rows = mat2d_rows(res);
	int cols = mat2d_cols(res);

	#pragma omp for nowait
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			mat2d_set(res, i, j, mat2d_get(res, i, j) + mat2d_get(m, i, j));
		}
	}
}

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

/**********************************************************
 * datatypes
 *********************************************************/
typedef struct __output_entry
{
	int index;
	double value;
} output_entry;

typedef struct __non_zero_entry
{
	int row;
	int col;
	double value;
} non_zero_entry;

typedef struct __dataset_info
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

typedef struct __grid_info
{
	int rows;
	int cols;
	int rows_periodic;
	int cols_periodic;
	int row;
	int col;
} grid_info;

typedef struct __input_info
{
	dataset_info dataset_info;
	non_zero_entry *entries;
} input_info;

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

/**********************************************************
 * mpiutil
 *********************************************************/
#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) \
			(BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(index,p,n) \
			(((p)*((index)+1)-1)/(n))

#define is_root(id) ((id) == 0)

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

void free_types(MPI_Datatype *nz, MPI_Datatype *data, MPI_Datatype *out) {
	MPI_Type_free(nz);
	MPI_Type_free(data);
	MPI_Type_free(out);
}

int smallest_divisor(int n) {
	if (n % 2 == 0)
		return 2;

	int div = 3;
	while (n % div != 0 && div <= n / div)
		div += 2;

	return div > n / div ? n : div;
}

void create_balanced_grid(const dataset_info *orig, int nproc, int *size, int dims)
{
	int sz[] = { 0, 0 };
	MPI_Dims_create(nproc, dims, sz);

	int items = orig->items;
	int users = orig->users;

	int ratio = items >= users ? items / users : users / items;

	if (ratio > 1) {
		int rows = sz[0];
		int cols = sz[1];
		int limit = nproc < ratio ? nproc : ratio;
		while (rows < limit) {
			int div = smallest_divisor(cols);
			cols /= div;
			rows *= div;
			if (rows > limit)
				break;
			sz[0] = rows;
			sz[1] = cols;
		}
	}

	/* swap grid coordinates */
	if (items > users) {
		int tmp = sz[0];
		sz[0] = sz[1];
		sz[1] = tmp;
	}

	size[0] = sz[0];
	size[1] = sz[1];
}

void create_cart_comm(MPI_Comm *comm, int *size, int dims)
{
	int periodic[dims];
	memset(periodic, 0, sizeof(int) * dims);
	MPI_Cart_create(MPI_COMM_WORLD, dims, size, periodic, 1, comm);
}

void split_comms(MPI_Comm cart_comm, MPI_Comm *row_comm, MPI_Comm *col_comm, int rank)
{
	int coords[2];
	MPI_Cart_coords(cart_comm, rank, 2, coords);
	MPI_Comm_split(cart_comm, coords[0], coords[1], row_comm);
	MPI_Comm_split(cart_comm, coords[1], coords[0], col_comm);
}

void free_comms(MPI_Comm *cart_comm, MPI_Comm *row_comm, MPI_Comm *col_comm) {
	MPI_Comm_free(cart_comm);
	MPI_Comm_free(row_comm);
	MPI_Comm_free(col_comm);
}

void free_ops(MPI_Op *reduce_op) {
	MPI_Op_free(reduce_op);
}

void print_non_zero_entries(int rank, non_zero_entry *entries, int size) {
	sleep(rank);
	printf("rank = %d : [ ", rank);
	for (int i = 0; i < size; i++) {
		printf("(%d, %d) ", entries[i].row, entries[i].col);
	}
	printf("]\n");
}

void max_cmp(output_entry *in, output_entry *inout, int *len, __attribute__((unused)) MPI_Datatype *type) {
	int sz = *len;
	for (int i = 0; i < sz; i++) {
		inout[i] = (in[i].value > inout[i].value ? in[i] : inout[i]);
	}
}

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

output_entry *compute_reduce_output(
	MPI_Comm row_comm,
	MPI_Datatype type,
	MPI_Op reduction_op,
	const dataset_info *local,
	const non_zero_entry *entries,
	const grid_info *grid,
	mat2d *L, mat2d *R)
{
	output_entry *red_res = NULL;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int row_rank;
	MPI_Comm_rank(row_comm, &row_rank);

	int usersz = local->users;
	int itemsz = local->items;

	int offset_row = BLOCK_LOW(grid->row, grid->rows, local->users_init);
	int offset_col = BLOCK_LOW(grid->col, grid->cols, local->items_init);

	if (is_root(row_rank)) {
		red_res = malloc(sizeof(output_entry) * usersz);
	}

	/* the array with the partial output */
	output_entry *out = malloc(sizeof(output_entry) * usersz);
	/* initialize the max array with a marker */
	for (int i = 0; i < usersz; i++) {
		out[i] = (output_entry) { -1, -1 };
	}

	for (int i = 0, aix = 0; i < usersz; i++) {
		output_entry max = { -1, -1 };
		for (int j = 0; j < itemsz; j++) {
			if (local->non_zero_sz == 0 || !(entries[aix].row == offset_row + i && entries[aix].col == offset_col + j)) {
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

	MPI_Reduce(out, red_res, usersz, type, reduction_op, 0, row_comm);

	free(out);

	return red_res;
}

output_entry *gather_output(
	int rank,
	MPI_Comm col_comm,
	MPI_Datatype type,
	const grid_info *grid,
	const output_entry *red_res,
	const dataset_info *local)
{
	output_entry *gathered_output = NULL;
	int *recvcounts = NULL;
	int *displs = NULL;

	int usersz = local->users;
	int rows = grid->rows;
	int cols = grid->cols;

	if (is_root(rank)) {
		/* the array where the partial outputs will be gathered in */
		gathered_output = malloc(sizeof(output_entry) * local->users_init);
		recvcounts = malloc(sizeof(int) * rows);
		displs = malloc(sizeof(int) * rows);

		for (int i = 0; i < rows; i++) {
			displs[i] = BLOCK_LOW(i, rows, local->users_init);
			recvcounts[i] = BLOCK_SIZE(i, rows, local->users_init);
		}
		memcpy(gathered_output, red_res, sizeof(output_entry) * usersz);
	}

	if (rank % cols == 0) {
		MPI_Gatherv(red_res, usersz, type, gathered_output, recvcounts, displs, type, 0, col_comm);
	}

	if (is_root(rank)) {
		free(recvcounts);
		free(displs);
	}

	return gathered_output;
}

void print_output(output_entry *gathered_out, int size) {
	for (int i = 0; i < size; i++) {
		int max = gathered_out[i].index;
		if (max != -1) {
			printf("%d\n", max);
		}
	}
}

void max_frontier(int *frontier, int reduce_L, const non_zero_entry *entries, int nz_size) {
	for (int f = *frontier ;
			f < nz_size - 1 && (reduce_L ? (entries[f].col == entries[f + 1].col) : (entries[f].row == entries[f + 1].row)) ;
			*frontier = ++f);
}

void next_frontier(int *frontier, int reduce_L, const non_zero_entry *entries, int nz_size) {
	for (int f = *frontier ;
			f < nz_size - 1 && f > 0 && (reduce_L ? (entries[f - 1].col == entries[f].col) : (entries[f - 1].row == entries[f].row)) ;
			*frontier = ++f);
}

void matrix_factorization(
		MPI_Comm row_comm,
		MPI_Comm col_comm,
		mat2d *L,
		mat2d *R,
		const dataset_info *info,
		non_zero_entry *entries,
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

	/* array with partial matrices */
	int reduce_L = (items > users);
	mat2d **reduction_array;

	if (reduce_L)
		qsort(entries, nz_size, sizeof(non_zero_entry), col_cmp);

	/* arrays with LOW and HIGH values for each thread */
	int *nz_lows;
	int *nz_highs;

	#pragma omp parallel
	{

	int num_threads = omp_get_num_threads();
	int tid = omp_get_thread_num();

	#pragma omp single
	{
		nz_lows = malloc(sizeof(int) * num_threads);
		nz_highs = malloc(sizeof(int) * num_threads);
		reduction_array = malloc(num_threads * sizeof(mat2d*));
	}

	if (reduce_L)
		reduction_array[tid] = mat2d_new(users, features);
	else
		reduction_array[tid] = mat2d_new(items, features);

	nz_lows[tid] = BLOCK_LOW(tid, num_threads, nz_size);
	nz_highs[tid] = BLOCK_HIGH(tid, num_threads, nz_size);

	#pragma omp barrier
	#pragma omp single
	{
		/*
		 * resolve low/high conflicts: maximize index without changing
		 * row/column for each thread as HIGH value
		 * if values of adjacent threads conflict,
		 * find first index of next row/column as LOW value for next thread
		 */
		max_frontier(&nz_highs[0], reduce_L, entries, nz_size);
		for (int i = 1; i < num_threads; i++) {
			max_frontier(&nz_highs[i], reduce_L, entries, nz_size);
			const non_zero_entry *prev_high = &entries[nz_highs[i - 1]];
			const non_zero_entry *curr_low = &entries[nz_lows[i]];
			if (reduce_L ? prev_high->col == curr_low->col : prev_high->row == curr_low->row) {
				next_frontier(&nz_lows[i], reduce_L, entries, nz_size);
			}
		}
	}

	int nz_low = nz_lows[tid];
	int nz_high = nz_highs[tid];

	for (int iter = 0; iter < iters; iter++)
	{
		is_root(col_rank) ? mat2d_copy_parallel(R, R_aux, tid, num_threads) : mat2d_zero_parallel(R_aux, tid, num_threads);
		is_root(row_rank) ? mat2d_copy_parallel(L, L_aux, tid, num_threads) : mat2d_zero_parallel(L_aux, tid, num_threads);

    	#pragma omp barrier

		mat2d *partial = reduction_array[tid];
		mat2d_zero(partial);

		/* no need for synchronization since each thread has a disjoint slice of non-zery entries */
		for (int n = nz_low; n <= nz_high; n++)
		{
			int i = entries[n].row - offset_row;
			int j = entries[n].col - offset_col;
			double value = alpha * 2 * (entries[n].value - mat2d_dot_product(L, i, R, j));

			int l_prod = i * features;
			int r_prod = j * features;

			for (int k = 0; k < features; k++) {
				int l_index = l_prod + k;
				int r_index = r_prod * features + k;

				if (reduce_L) {
					mat2d_set_index(R_aux, r_index, mat2d_get_index(R_aux, r_index) - value *
						(-mat2d_get_index(L, l_index)));

					mat2d_set_index(partial, l_index, mat2d_get_index(partial, l_index) - value *
						(-mat2d_get_index(R, r_index)));
				} else {
					mat2d_set_index(L_aux, l_index, mat2d_get_index(L_aux, l_index) - value *
						(-mat2d_get_index(R, r_index)));

					mat2d_set_index(partial, r_index, mat2d_get_index(partial, r_index) - value *
						(-mat2d_get_index(L, l_index)));
				}
			}
		}
		#pragma omp barrier

		mat2d *to_reduce = (reduce_L ? L_aux : R_aux);
		for (int t = 0; t < num_threads; t++) {
			mat2d_sum(to_reduce, reduction_array[t]);
		}

		#pragma omp barrier
		#pragma omp sections
		{
			#pragma omp section
			MPI_Allreduce(mat2d_data(L_aux), mat2d_data(L), mat2d_size(L), MPI_DOUBLE, MPI_SUM, row_comm);
			#pragma omp section
			MPI_Allreduce(mat2d_data(R_aux), mat2d_data(R), mat2d_size(R), MPI_DOUBLE, MPI_SUM, col_comm);
		}
	}

	free(reduction_array[tid]);

	}

	free(reduction_array);
	free(nz_lows);
	free(nz_highs);

	if (reduce_L)
		qsort(entries, nz_size, sizeof(non_zero_entry), row_cmp);

	mat2d_free(L_aux);
	mat2d_free(R_aux);
}

/**
 * Read non-zero entries to file according to some restrictions
 * returns -1 if error otherwise returns number of entries read
 */
int read_non_zero_entries(
	FILE *fp, /* file from where to read next entries */
	non_zero_entry *buffer, /* buffer where to write read non-zero entries */
	int buffsz, /* max number of entries read */
	int blk_high, /*up to which row to read */
	int *__next_row /* the next row in the file, before rewinding */)
{
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
			*__next_row = -1;
			return entries_read;
		}
		if (elems_read != 3) {
			return -1;
		}

		/* rewind read line when row exceeds HIGH value */
		if (row > blk_high) {
			fseek(fp, ftell_before - ftell(fp), SEEK_CUR);
			*__next_row = row;
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
		input_info *local,
		const dataset_info *orig,
		const grid_info *grid)
{

	int users = orig->users;
	int items = orig->items;
	/**
	 * temporary buffer to store read non-zero entries
	 * size must be the maximum number of non-zero elements between the lines of a decomposed block
	 */
	int tmpsize = items;
	non_zero_entry *tmpbuffer = malloc(sizeof(non_zero_entry) * tmpsize);

	int grid_cols = grid->cols;
	int grid_rows = grid->rows;

	/* temporary non-zery entry array the root will accumulate read values into */
	non_zero_entry *tmp_nz_entries = NULL;

	/**
	 * if no non-zero entries in grid block, send something back to
	 * avoid the receiving process staying blocked forever
	 * save in this array which ranks have been sent to
	 */
	int nprocs;
	MPI_Comm_size(cart_comm, &nprocs);
	int order_sent[nprocs];
	memset(order_sent, 0, nprocs * sizeof(int));

	for (int row = 0, next_row; row < users && row >= 0; row = next_row) {

		int entries_read = read_non_zero_entries(fp, tmpbuffer, tmpsize, row, &next_row);
		if (entries_read < 0) {
			die("Unable to read non-zero entries");
		}

		int grid_row = BLOCK_OWNER(row, grid_rows, users);

		non_zero_entry *base = tmpbuffer;
		for (int i = 0; i < entries_read; i++) {

			int grid_col = BLOCK_OWNER(tmpbuffer[i].col, grid_cols, items);

			if (grid_col > grid_cols) {
				die("More columns read than grid width");
			}

			non_zero_entry *frontier;
			if (i < entries_read - 1)
				frontier = &tmpbuffer[i + 1];
			else
				frontier = &tmpbuffer[i];

			int rank;
			int coords[] = { grid_row, grid_col };
			MPI_Cart_rank(cart_comm, coords, &rank);

			if (i == entries_read - 1 || frontier->col > BLOCK_HIGH(grid_col, grid_cols, orig->items)) {

				/**
				 * if it is the last entry in the buffer
				 * need to consider one more value to have correct size
				 */
				size_t offset = frontier - base + (i == entries_read - 1);

				/* if first time sending to rank */
				if (order_sent[rank] == 0) {

					int blk_sz_users = BLOCK_SIZE(grid_row, grid_rows, orig->users);
					int blk_sz_items = BLOCK_SIZE(grid_col, grid_cols, orig->items);

					dataset_info tmpinfo = (dataset_info) {
						orig->iters,
						orig->features,
						blk_sz_users,
						blk_sz_items,
						orig->users,
						orig->items,
						offset,
						orig->alpha };

					if (is_root(rank)) {
						local->dataset_info = tmpinfo;
						if (tmp_nz_entries == NULL)
							tmp_nz_entries = malloc(sizeof(non_zero_entry) * blk_sz_users * blk_sz_items);
						memcpy(tmp_nz_entries, base, sizeof(non_zero_entry) * offset);
					} else {
						MPI_Send(&tmpinfo, 1, dataset_info_type, rank, 0, MPI_COMM_WORLD);
						MPI_Send(base, offset, non_zero_type, rank, 1, MPI_COMM_WORLD);
					}

				} else if (is_root(rank)) {
					non_zero_entry *dest = &tmp_nz_entries[local->dataset_info.non_zero_sz];
					memcpy(dest, base, sizeof(non_zero_entry) * offset);
					local->dataset_info.non_zero_sz += offset;
				} else {
					MPI_Send(&offset, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
					MPI_Send(base, offset, non_zero_type, rank, 1, MPI_COMM_WORLD);
				}

				order_sent[rank]++;
				base = frontier;
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

	for (int r = 0; r < nprocs; r++) {

		if (order_sent[r] == 0) {
			int coords[2];
			MPI_Cart_coords(cart_comm, r, 2, coords);
			ds_info_to_send.users = BLOCK_SIZE(coords[0] /*row*/, grid_rows, orig->users);
			ds_info_to_send.items = BLOCK_SIZE(coords[1] /*col*/, grid_cols, orig->items);
			if (is_root(r)) {
				local->dataset_info = ds_info_to_send;
				local->entries = NULL;
			} else {
				MPI_Send(&ds_info_to_send, 1, dataset_info_type, r, 0, MPI_COMM_WORLD);
			}
		} else if (order_sent[r] < users) {
			int marker = 0;
			MPI_Send(&marker, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
		}
	}

	/* trim non-zery entries to size */
	int non_zero_sz = local->dataset_info.non_zero_sz;
	if (non_zero_sz > 0) {
		local->entries = malloc(sizeof(non_zero_entry) * non_zero_sz);
		memcpy(local->entries, tmp_nz_entries, sizeof(non_zero_entry) * non_zero_sz);
	}
	free(tmp_nz_entries);
}

void receive_non_zero_values(
		input_info *local,
		MPI_Datatype ds_info_type,
		MPI_Datatype nz_type,
		MPI_Status *status)
{
	/* receive dataset info (first send) */
	MPI_Recv(&local->dataset_info, 1, ds_info_type, 0, 0, MPI_COMM_WORLD, status);

	/* if size is zero it means there are no non-zero entries to receive for this rank, return */
	int non_zero_size = local->dataset_info.non_zero_sz;
	if (non_zero_size == 0) {
		local->entries = NULL;
		return;
	}

	int users = local->dataset_info.users_init;

	/* alloc maximum size for non-zero entries */
	non_zero_entry *tmp_entries = malloc(sizeof(non_zero_entry) * local->dataset_info.users * local->dataset_info.items);
	non_zero_entry *base = tmp_entries;

	/* receive first non-zero entries chunk (second send) */
	MPI_Recv(base, non_zero_size, nz_type, 0, 1, MPI_COMM_WORLD, status);

	/* this loop may do up to users - 1iterations if all ranks have non-zero entries for each line */
	for (int i = 0; i < users - 1; i++) {
		base += non_zero_size;
		MPI_Recv(&non_zero_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, status);
		if (non_zero_size == 0)
			break;
		MPI_Recv(base, non_zero_size, nz_type, 0, 1, MPI_COMM_WORLD, status);
		local->dataset_info.non_zero_sz += non_zero_size;
	}

	/* trim non-zery entries to size */
	non_zero_size = local->dataset_info.non_zero_sz;
	local->entries = malloc(sizeof(non_zero_entry) * non_zero_size);
	memcpy(local->entries, tmp_entries, sizeof(non_zero_entry) * non_zero_size);
	free(tmp_entries);

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

	MPI_Status status;

	MPI_Datatype non_zero_type;
	MPI_Datatype dataset_info_type;

	int provided_level;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_level);

	if (provided_level != MPI_THREAD_MULTIPLE) {
		die("Unsupported provided level");
	}

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	create_non_zero_entry(&non_zero_type);
	create_dataset_info(&dataset_info_type);

	/* the original, not partitioned dataset information */
	dataset_info orig;

	if (is_root(rank)) {
		if (read_input_metadata(fp, &orig) != 0) {
			die("Unable to initialize parameters.");
		}
		orig.users_init = orig.users;
		orig.items_init = orig.items;
	}

	/* broadcast the original dataset info to know how to balance the grid */
	MPI_Bcast(&orig, 1, dataset_info_type, 0, MPI_COMM_WORLD);

	MPI_Comm cart_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;

	int size[2];
	create_balanced_grid(&orig, nproc, size, 2);
	create_cart_comm(&cart_comm, size, 2);
	MPI_Comm_rank(cart_comm, &rank);

	split_comms(cart_comm, &row_comm, &col_comm, rank);
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_rank(col_comm, &col_rank);

	grid_info grid;
	MPI_Cart_get(cart_comm, 2, &grid.rows, &grid.rows_periodic, &grid.row);

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
		distribute_non_zero_values(fp,
			cart_comm, non_zero_type, dataset_info_type,
			&local, &orig, &grid);

		if (fclose(fp) == EOF)
			fprintf(stderr, "Unable to close file");
	} else {
		receive_non_zero_values(&local,
			dataset_info_type,
			non_zero_type,
			&status);
	}

	L = mat2d_new(local.dataset_info.users, local.dataset_info.features);

	if (is_root(rank)) {
		mat2d_random_fill(L, local.dataset_info.features);
		if (grid.rows > 1)
			distribute_matrix_L(cart_comm, grid.rows, orig.users, orig.features);
	} else if (grid.rows > 1 && rank % grid.cols == 0) {
		MPI_Recv(mat2d_data(L), mat2d_size(L), MPI_DOUBLE, 0, 2, cart_comm, &status);
	}

	MPI_Bcast(mat2d_data(L), mat2d_size(L), MPI_DOUBLE, 0, row_comm);

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

	matrix_factorization(row_comm, col_comm, L, R, &local.dataset_info, local.entries, &grid);

	MPI_Datatype output_entry_type;
	create_output_entry(&output_entry_type);

	MPI_Op reduce_output_op;
	MPI_Op_create((MPI_User_function *) max_cmp, 1, &reduce_output_op);

	output_entry *reduction_result = compute_reduce_output(
		row_comm, output_entry_type, reduce_output_op,
		&local.dataset_info, local.entries, &grid, L, R);

	free(local.entries);
	mat2d_free(L);
	mat2d_free(R);

	output_entry *gathered_output = gather_output(
		rank, col_comm, output_entry_type,
		&grid, reduction_result, &local.dataset_info);

	free(reduction_result);

	if (is_root(rank)) {
		print_output(gathered_output, local.dataset_info.users_init);
	}

	free(gathered_output);

	free_types(&non_zero_type, &dataset_info_type, &output_entry_type);
	free_comms(&cart_comm, &row_comm, &col_comm);
	free_ops(&reduce_output_op);

	MPI_Finalize();

	return 0;
}
