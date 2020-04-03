# the compiler
CC = gcc

# compiler flags
CFLAGS = -Wall -Wextra

# output name
OUT = matFact

default: serial

serial: matFact.o
	$(CC) $(CFLAGS) -o $(OUT) matFact.o

matFact.o: matFact.c
	$(CC) $(CFLAGS) -c matFact.c

clean:
	rm -f $(OUT) *.o