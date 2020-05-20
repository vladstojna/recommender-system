# the compiler
MPICC = mpicc

# compiler flags
CFLAGS = -Wall -Wextra -fopenmp

# output name
OUT = matFact-mpi.out

default: mpi

mpi: matFact-mpi.o
	$(MPICC) $(CFLAGS) -o $(OUT) matFact-mpi.o

hybrid: matFact-hybrid.o
	$(MPICC) $(CFLAGS) -o $(OUT) matFact-hybrid.o

matFact-mpi.o: matFact-mpi.c
	$(MPICC) $(CFLAGS) -c matFact-mpi.c

matFact-hybrid.o: matFact-hybrid.c
	$(MPICC) $(CFLAGS) -c matFact-hybrid.c

clean:
	rm -f $(OUT) *.o
