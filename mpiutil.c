// mpi utils

#include "mpiutil.h"
#include "datatypes.h"

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
