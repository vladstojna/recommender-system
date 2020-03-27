#include <stdio.h>
#include <stdlib.h>

#include "adjlst.h"

adj_lst *adjlst_new(size_t rows) {
	adj_lst *result = malloc(sizeof(adj_lst));
	if (result) {
		result->rows = rows;
		result->columns = calloc(rows, sizeof(adj_lst_entry));
		if (!result->columns) {
			free(result);
			return NULL;
		}
	}
	return result;
}

void adjlst_free(adj_lst *lst) {
	if (!lst) {
		return;
	}
	if (!lst->columns) {
		free(lst);
		return;
	}
	for (size_t i = 0; i < lst->rows; ++i) {
		adj_lst_entry *columns = &lst->columns[i];
		if (columns) {
			free(columns->entries);
		}
	}
	free(lst->columns);
	free(lst);
}

column_entry *adjlst_new_entries(size_t cols) {
	return calloc(cols, sizeof(column_entry));
}

void adjlst_print(adj_lst *lst) {
	for (size_t i = 0; i < lst->rows; ++i) {
		printf("%4ld | ", i);
		for (size_t j = 0; j < lst->columns[i].size; ++j) {
			printf("[%4d | %4f] ", lst->columns[i].entries[j].at, lst->columns[i].entries[j].value);
		}
		printf("\n");
	}
}
