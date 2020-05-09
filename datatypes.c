// datatypes

#include "datatypes.h"

#include <stdio.h>

int col_cmp(const void *a, const void *b) {
	non_zero_entry *nz_a = (non_zero_entry*) a;
	non_zero_entry *nz_b = (non_zero_entry*) b;
	return nz_a->col == nz_b->col ? nz_a->row - nz_b->row : nz_a->col - nz_b->col;
}

int row_cmp(const void *a, const void *b) {
	non_zero_entry *nz_a = (non_zero_entry*) a;
	non_zero_entry *nz_b = (non_zero_entry*) b;
	return nz_a->row == nz_b->row ? nz_a->col - nz_b->col : nz_a->row - nz_b->row;
}

void print_dataset_info(int rank, dataset_info *info) {
	printf("----- Rank=%d Iters=%d Features=%d Users=%d(%d) Items=%d(%d) Non-zero size=%d Alpha=%f -----\n",
		rank, info->iters, info->features,
		info->users, info->users_init, info->items,
		info->items_init, info->non_zero_sz, info->alpha);
}
