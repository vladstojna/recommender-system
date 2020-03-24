#ifndef ADJLST_H
#define ADJLST_H

#include <stdlib.h>

typedef struct {
	int at;
	double value;
} column_entry;

typedef struct {
	size_t size;
	column_entry *entries;
} adj_lst_entry;

typedef struct {
	size_t rows;
	adj_lst_entry *columns;
} adj_lst;

adj_lst *adjlst_new(size_t rows);
column_entry *adjlst_new_entries(size_t cols);
void adjlst_free(adj_lst *lst);
void adjlst_print(adj_lst *lst);

#define adjlst_entries(al, r) ((al)->columns[r].entries)
#define adjlst_entries_sz(al, r) ((al)->column[r].size)
#define adjlst_entries_set(al, r, e, sz) { (al)->columns[r].entries = e; (al)->columns[r].size = sz; }
#define adjlst_is_empty(al, r) ((al)->columns[r].size == 0)

#endif
