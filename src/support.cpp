#include "support.h"

char *do_file_name(char *path, char *type, char *label, char *who, char *extension)
{
    uint path_len = strlen(path);
    uint type_len = strlen(type);
    uint label_len = strlen(label);
    uint who_len = strlen(who);
    uint extension_len = strlen(extension);
    uint length = path_len + type_len + label_len + who_len
                        + extension_len + 5;
    char *filename = new char [length];

    strcpy(filename, path);
    filename[path_len] = '/';
    strcpy(&filename[path_len + 1], type);
    filename[path_len + 1 + type_len] = '_';
    strcpy(&filename[path_len + 2 + type_len], label);
    filename[path_len + 2 + type_len + label_len] = '_';
    strcpy(&filename[path_len + 3 + type_len + label_len], who);
    filename[path_len + 3 + type_len + label_len + who_len] = '.';
    strcpy(&filename[path_len + 4 + type_len + label_len + who_len], extension);
    filename[length] = '\0';

    for (uint i = 0; i < length; i++)
        if (isspace(filename[i]))
            filename[i] = '_';
    return filename;
}


char *rank_to_string(int rank, int size)
{
    uint len_size, len_rank;
    len_size = len_rank = 0;
    while (size) {
        size /= 10;
        len_size++;
    }
    if (rank) {
        int tmp_rank = rank;
        while (tmp_rank) {
            tmp_rank /= 10;
            len_rank++;
        } 
    } else {
        len_rank = 1;
    }
    char *str_rank = new char [len_size + 1];
    for (uint i = 0; i < len_size; i++)
        str_rank[i] = '0';
    int i = 1;
    while (rank) {
        str_rank[len_size - i] = rank % 10 + '0';
        rank /= 10;
        i++;
    }
    //sprintf(&str_rank[len_size - len_rank], "%d", rank);
    str_rank[len_size] = '\0';
    return str_rank;
}
