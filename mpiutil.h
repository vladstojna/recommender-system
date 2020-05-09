#ifndef __MPIUTIL_H__
#define __MPIUTIL_H__

#include <mpi.h>

#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) \
			(BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(index,p,n) \
			(((p)*((index)+1)-1)/(n))

#define is_root(id) ((id) == 0)

int create_non_zero_entry(MPI_Datatype *type);
int create_dataset_info(MPI_Datatype *type);
int create_output_entry(MPI_Datatype *type);
void create_cart_comm(MPI_Comm *comm, int nproc);
void split_comms(MPI_Comm cart_comm, MPI_Comm *row_comm, MPI_Comm *col_comm, int rank);

#endif
