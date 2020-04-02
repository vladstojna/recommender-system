#ifndef MAT2D_H
#define MAT2D_H

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef struct
{
	int n_r;
	int n_c;
	double *data;
} mat2d;

mat2d *mat2d_new(int rows, int cols);
void mat2d_free(mat2d *mat);

double *mat2d_get_line(mat2d *mat, int line);
void mat2d_set_line(mat2d *mat, int line, double *value);

void mat2d_random_fill_LR(mat2d *L, mat2d *R, double norm);
void mat2d_copy(mat2d *from, mat2d *to);
void mat2d_copy_parallel(mat2d *from, mat2d *to, int tid, int num_threads);
void mat2d_zero(mat2d *mat);
void mat2d_zero_parallel(mat2d *mat, int tid, int num_threads);

void mat2d_print(mat2d *mat);

void mat2d_sum(mat2d *res, mat2d *m);
void mat2d_prod(mat2d *left, mat2d *right, mat2d *dest);
void mat2d_transpose(mat2d *orig, mat2d *transpose);
double mat2d_dot_product(mat2d *left, int r, mat2d *right, int c);

#define mat2d_rows(m) m->n_r
#define mat2d_cols(m) m->n_c
#define mat2d_get(m, r, c) m->data[((r)*m->n_c) + (c)]
#define mat2d_set(m, r, c, v) m->data[((r)*m->n_c) + (c)] = v

#endif