#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>

// Print error message and kill program
void die(const char *error);

// Parse int from file
void parse_int(FILE *fp, int *i);

// Parse double from file
void parse_double(FILE *fp, double *d);

// Parse three ints from file
void parse_three_ints(FILE *fp, int *a, int *b, int *c);

// Parse non-zero entry from file
void parse_non_zero_entry(FILE *fp, int *row, int *col, double *value);

#endif