#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>

// Print error message and kill program
void die(const char *error);

// Parse int from file
int parse_int(FILE *fp);

// Parse double from file
double parse_double(FILE *fp);

#endif