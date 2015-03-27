#include "dataMPI.h"

template <class DataType, class LengthData>
DataMPI<DataType, LengthData>::DataMPI(MyMPI new_me)
    : data(0), length(0), me(new_me)
{
}

template <class DataType, class LengthData>
DataMPI<DataType, LengthData>::~DataMPI()
{
}


template <class DataType, class LengthData>
bool DataMPI<DataType, LengthData>::isEmpty()
{
    return data ? false : true;
}

template <class DataType, class LengthData>
void DataMPI<DataType, LengthData>::readFile(char *file_name)
{
    readMPI(file_name);
}

template <class DataType, class LengthData>
void DataMPI<DataType, LengthData>::writeFile(char *file_name)
{
    writeMPI(file_name);
}

template <class DataType, class LengthData>
void DataMPI<DataType, LengthData>::free()
{
    delete [] data;
}


template <class DataType, class LengthData>
LengthData DataMPI<DataType, LengthData>::offsetLength(LengthData* &offset, LengthData* &sum_offset, LengthData *var)
{
    offset = new LengthData [me.getSize()];
    sum_offset = new LengthData [me.getSize()];
    me.Allgather(var, 1, MPI_UNSIGNED_LONG, offset, 1, MPI_UNSIGNED_LONG);
    LengthData sum_length = 0;
    for (int i = 0; i < me.getSize(); i++) {
        sum_offset[i] = sum_length;
        sum_length += offset[i];
    }
    return sum_length;
}
