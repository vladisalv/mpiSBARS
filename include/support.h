#ifndef __SUPPORT_HEADER__
#define __SUPPORT_HEADER__

#include "types.h"

#include <string.h>
#include <ctype.h>

char *do_file_name(char *path, char *type, char *label, char *who, char *extension);
char *rank_to_string(int rank, int size);

#endif /* __SUPPORT_HEADER__ */
