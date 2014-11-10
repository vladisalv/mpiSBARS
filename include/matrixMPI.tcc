#include "matrixMPI.h"

template <class DataType, class LengthData>
MatrixMPI<DataType, LengthData>::MatrixMPI(MyMPI me)
    : DataMPI<DataType, LengthData>(me), width(0), height(0)
{
}

template <class DataType, class LengthData>
MatrixMPI<DataType, LengthData>::~MatrixMPI()
{
}


template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::readMPI(char *file_name)
{
}

template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::readUsually(char *file_name)
{
    ;
}

template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::readMy(char *file_name)
{
    ;
}


template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::writeMPI(char *file_name)
{
}

template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::writeUsually(char *file_name)
{
}

template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::writeMy(char *file_name)
{
    ;
}
