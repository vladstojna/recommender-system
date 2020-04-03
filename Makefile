# the compiler
CC = gcc

# compiler flags
CFLAGS = -Wall -Wextra -fopenmp

# output name
OUT = matFact.out

default: serial

serial: matFact.o mat2d.o util.o
	$(CC) $(CFLAGS) -o $(OUT) matFact.o mat2d.o util.o

omp: matFact-omp.o mat2d.o util.o
	$(CC) $(CFLAGS) -o $(OUT) matFact-omp.o mat2d.o util.o

matFact.o: matFact.c
	$(CC) $(CFLAGS) -c matFact.c

matFact-omp.o: matFact-omp.c
	$(CC) $(CFLAGS) -c matFact-omp.c

mat2d.o: mat2d.c mat2d.h util.h
	$(CC) $(CFLAGS) -c mat2d.c

util.o: util.c util.h
	$(CC) $(CFLAGS) -c util.c

clean:
	rm -f $(OUT) *.o
