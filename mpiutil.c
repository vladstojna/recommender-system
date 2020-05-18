// mpi utils

#include "mpiutil.h"
#include "datatypes.h"

#include <string.h>

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
	int rows = sz[0];
	int cols = sz[1];

	if (ratio > 1) {
		int div = smallest_divisor(cols);
		int limit = nproc < ratio ? nproc : ratio;
		while (rows <= limit) {
			sz[0] = rows;
			sz[1] = cols;
			cols /= div;
			rows *= div;
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
