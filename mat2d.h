#ifndef MAT2D_H
#define MAT2D_H

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct {
	int n_r;
	int n_c;
	double* data;
} mat2d;

mat2d* mat2d_new(int rows, int cols);
void mat2d_free(mat2d* mat);
void mat2d_copy(mat2d* from, mat2d* to);
void mat2d_print(mat2d* mat);
void mat2d_random_fill_LR(mat2d* L, mat2d* R, double norm);
void mat2d_prod(mat2d* left, mat2d* right, mat2d* dest);
mat2d* mat2d_transpose(mat2d* orig);

#define mat2d_get(m, r, c) m->data[((r)*m->n_c)+(c)]
#define mat2d_set(m, r, c, v) m->data[((r)*m->n_c)+(c)]=v

#endif