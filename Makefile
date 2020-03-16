# the compiler
CC = gcc

# compiler flags
CFLAGS = -Wall -Wextra -g
OPT = -O3

# output name
OUT = matFact.out

default: serial

serial: matFact.o mat2.o util.o
	$(CC) $(CFLAGS) -o $(OUT) matFact.o mat2.o util.o

serial-opt: matFact.o mat2.o util.o
	$(CC) $(CFLAGS) $(OPT) -o $(OUT) matFact.o mat2.o util.o

omp: matFact-omp.o
	$(CC) $(CFLAGS) -o $(OUT) matFact-omp.o

omp-opt: matFact-omp.o
	$(CC) $(CFLAGS) $(OPT) -o $(OUT) matFact-omp.o

matFact.o: matFact.c
	$(CC) $(CFLAGS) -c matFact.c

matFact-omp.o: matFact-omp.c
	$(CC) $(CFLAGS) -c matFact-omp.c

mat2.o: mat2.c mat2.h util.h
	$(CC) $(CFLAGS) -c mat2.c

util.o: util.c util.h
	$(CC) $(CFLAGS) -c util.c

clean:
	rm -f $(OUT) *.o
