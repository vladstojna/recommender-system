#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>

// Print error message and kill program
void die(const char* error);

// Parse int from string str
int parse_int(FILE* str);

// Parse double from string str
double parse_double(FILE* str);

#endif