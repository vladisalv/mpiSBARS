#include "dataMPI.h"

template <class DataType>
DataMPI<DataType>::DataMPI(MyMPI new_me)
    : data(0), length(0), me(new_me)
{
}

template <class DataType>
DataMPI<DataType>::~DataMPI()
{
}


template <class DataType>
bool DataMPI<DataType>::isEmpty()
{
    return data ? false : true;
}

template <class DataType>
void DataMPI<DataType>::readFile(char *file_name)
{
    readMPI(file_name);
}

template <class DataType>
void DataMPI<DataType>::writeFile(char *file_name)
{
    writeMPI(file_name);
}

template <class DataType>
void DataMPI<DataType>::free()
{
    delete [] data;
    data = 0;
}


template <class DataType>
ulong DataMPI<DataType>::offsetLength(ulong* &offset, ulong* &sum_offset, ulong *var)
{
    offset = new ulong [me.getSize()];
    sum_offset = new ulong [me.getSize()];
    me.Allgather(var, 1, MPI_UNSIGNED_LONG, offset, 1, MPI_UNSIGNED_LONG);
    ulong sum_length = 0;
    for (int i = 0; i < me.getSize(); i++) {
        sum_offset[i] = sum_length;
        sum_length += offset[i];
    }
    return sum_length;
}

template <class DataType>
inline MPI_Datatype DataMPI<DataType>::getMpiDataType()
{
    return MPI_BYTE;
}
/*
template <>
inline MPI_Datatype DataMPI::getMpiDataType<char>()
{
    return MPI_CHAR;
}

template <>
inline MPI_Datatype DataMPI::getMpiDataType<int>()
{
    return MPI_CHAR;
}

template <>
inline MPI_Datatype DataMPI::getMpiDataType<uint>()
{
    return MPI_CHAR;
}

template <>
inline MPI_Datatype DataMPI::getMpiDataType<float>()
{
    return MPI_CHAR;
}

template <>
inline MPI_Datatype DataMPI::getMpiDataType<double>()
{
    return MPI_CHAR;
}

template <>
inline MPI_Datatype DataMPI::getMpiDataType<bool>()
{
    return MPI_CHAR;
}
*/

/*
template <>
inline MPI_Datatype getMpiDataType<Sequence>()
{
    return MPI_CHAR;
}

template <>
inline MPI_Datatype getMpiDataType<Profile>()
{
    return MPI_CHAR;
}
*/
