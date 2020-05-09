#ifndef __DATATYPES_H__
#define __DATATYPES_H__

typedef struct __output_entry
{
	int index;
	double value;
} output_entry;

typedef struct __non_zero_entry
{
	int row;
	int col;
	double value;
} non_zero_entry;

typedef struct __dataset_info
{
	int iters;
	int features;
	int users;
	int items;
	int users_init;
	int items_init;
	int non_zero_sz;
	double alpha;
} dataset_info;

typedef struct __grid_info
{
	int rows;
	int cols;
	int rows_periodic;
	int cols_periodic;
	int row;
	int col;
} grid_info;

typedef struct __input_info
{
	dataset_info dataset_info;
	non_zero_entry *entries;
} input_info;

int col_cmp(const void *a, const void *b);
int row_cmp(const void *a, const void *b);
void print_dataset_info(int rank, dataset_info *info);

#endif
