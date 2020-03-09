# the compiler
CC = gcc

# compiler flags
CFLAGS = -Wall -Wextra -g
OPT = -O3

# output name
OUT = matFact.out

default: serial

serial: matFact.o
	$(CC) $(CFLAGS) -o $(OUT) matFact.o

serial-opt: matFact.o
	$(CC) $(CFLAGS) $(OPT) -o $(OUT) matFact.o

omp: matFact-omp.o
	$(CC) $(CFLAGS) -o $(OUT) matFact-omp.o

omp-opt: matFact-omp.o
	$(CC) $(CFLAGS) $(OPT) -o $(OUT) matFact-omp.o

matFact.o: matFact.c
	$(CC) $(CFLAGS) -c matFact.c

matFact-omp.o: matFact-omp.c
	$(CC) $(CFLAGS) -c matFact-omp.c

clean:
	rm -f $(OUT) *.o
