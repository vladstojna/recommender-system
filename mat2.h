#ifndef MAT2_H
#define MAT2_H

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct {
	int n_r;
	int n_c;
	double* data;
} mat2;

mat2* mat2_new(int rows, int cols);
void mat2_free(mat2* mat);
void mat2_copy(mat2* from, mat2* to);
void mat2_print(mat2* mat);
void mat2_random_fill(mat2* mat, double norm);
void mat2_prod(mat2* left, mat2* right, mat2* dest);
mat2* mat2_transpose(mat2* orig);

#define mat2_get(m, r, c) m->data[((r)*m->n_c)+(c)]
#define mat2_set(m, r, c, v) m->data[((r)*m->n_c)+(c)]=v

#endif