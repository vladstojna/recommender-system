# the compiler
CC = gcc
MPICC = mpicc

# compiler flags
CFLAGS = -Wall -Wextra -fopenmp

# output name
OUT = matFact.out

default: serial

serial: matFact.o mat2d.o util.o datatypes.o
	$(CC) $(CFLAGS) -o $(OUT) matFact.o mat2d.o util.o datatypes.o

omp: matFact-omp.o mat2d.o util.o datatypes.o
	$(CC) $(CFLAGS) -o $(OUT) matFact-omp.o mat2d.o util.o datatypes.o

mpi: matFact-mpi.o mat2d.o util.o datatypes.o mpiutil.o
	$(MPICC) $(CFLAGS) -o $(OUT) matFact-mpi.o mat2d.o util.o datatypes.o mpiutil.o

matFact.o: matFact.c
	$(CC) $(CFLAGS) -c matFact.c

matFact-omp.o: matFact-omp.c
	$(CC) $(CFLAGS) -c matFact-omp.c

matFact-mpi.o: matFact-mpi.c
	$(MPICC) $(CFLAGS) -c matFact-mpi.c

mat2d.o: mat2d.c mat2d.h util.h
	$(CC) $(CFLAGS) -c mat2d.c

util.o: util.c util.h
	$(CC) $(CFLAGS) -c util.c

mpiutil.o: mpiutil.c mpiutil.h
	$(MPICC) $(CFLAGS) -c mpiutil.c

datatypes.o: datatypes.c datatypes.h
	$(CC) $(CFLAGS) -c datatypes.c

clean:
	rm -f $(OUT) *.o
