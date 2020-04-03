# the compiler
CC = gcc

# compiler flags
CFLAGS = -Wall -Wextra -fopenmp

# output name
OUT = matFact.out

default: omp

omp: matFact-omp.o
	$(CC) $(CFLAGS) -o $(OUT) matFact-omp.o

matFact-omp.o: matFact-omp.c
	$(CC) $(CFLAGS) -c matFact-omp.c

clean:
	rm -f $(OUT) *.o
