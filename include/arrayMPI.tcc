#include "arrayMPI.h"

template <class DataType, class LengthData>
ArrayMPI<DataType, LengthData>::ArrayMPI(MyMPI me)
    : DataMPI<DataType, LengthData>(me)
{
}

template <class DataType, class LengthData>
ArrayMPI<DataType, LengthData>::~ArrayMPI()
{
}

template <class DataType, class LengthData>
void ArrayMPI<DataType, LengthData>::readMPI(char *file_name)
{
    MPI_File fh;
    MPI_Offset length_file;
    fh = this->me.openFile(file_name, MPI_MODE_RDONLY, MPI_INFO_NULL); // MPI_Open_file

    length_file = this->me.getSizeFile(fh);
    this->length = length_file / this->me.getSize();
    MPI_Offset offset = this->length * this->me.getRank();
    if (this->me.isLast())
        this->length += length_file % this->me.getSize();

    delete [] this->data; // if you forget about old data
    this->data = new DataType [this->length];
    this->me.readFile(fh, offset, this->data, this->length, MPI_CHAR, MPI_INFO_NULL);
    this->me.closeFile(&fh);
}

template <class DataType, class LengthData>
void ArrayMPI<DataType, LengthData>::readUsually(char *file_name)
{
}

template <class DataType, class LengthData>
void ArrayMPI<DataType, LengthData>::readMy(char *file_name)
{
}

template <class DataType, class LengthData>
void ArrayMPI<DataType, LengthData>::writeMPI(char *file_name)
{
}

template <class DataType, class LengthData>
void ArrayMPI<DataType, LengthData>::writeUsually(char *file_name)
{
}

template <class DataType, class LengthData>
void ArrayMPI<DataType, LengthData>::writeMy(char *file_name)
{
}
