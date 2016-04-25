#include "arrayMPI.h"

template <class DataType>
ArrayMPI<DataType>::ArrayMPI(MyMPI me)
    : DataMPI<DataType>(me), offset(0)
{
}

template <class DataType>
ArrayMPI<DataType>::~ArrayMPI()
{
}

template <class DataType>
void ArrayMPI<DataType>::readMPI(char *file_name)
{
#ifdef USE_MPI
    MPI_File fh;
    MPI_Offset length_file;
    fh = this->me.openFile(file_name, MPI_MODE_RDONLY, MPI_INFO_NULL); // MPI_Open_file

    length_file = this->me.getSizeFile(fh);
    this->length = length_file / this->me.getSize();
    MPI_Offset offset = this->length * this->me.getRank();
    this->offset = offset; // ??? it size_t in byte. for other class (not sequence) it can be not
    if (this->me.isLast())
        this->length += length_file % this->me.getSize();

    delete [] this->data; // if you forget about old data
    this->data = new DataType [this->length];
    this->me.readFile(fh, offset, this->data, this->length, this->getMpiDataType(), MPI_INFO_NULL);
    this->me.closeFile(&fh);
#endif
}

template <class DataType>
void ArrayMPI<DataType>::readUsually(char *file_name)
{
}

template <class DataType>
void ArrayMPI<DataType>::readMy(char *file_name)
{
}

template <class DataType>
void ArrayMPI<DataType>::writeMPI(char *file_name)
{
#ifdef USE_MPI
#endif
}

template <class DataType>
void ArrayMPI<DataType>::writeUsually(char *file_name)
{
}

template <class DataType>
void ArrayMPI<DataType>::writeMy(char *file_name)
{
}


template <class DataType>
void ArrayMPI<DataType>::debugInfo(const char *file, int line, const char *info)
{
    this->me.rootMessage("\n");
    this->me.rootMessage("This is debugInfo(%s) in %s at line %d\n", info, file, line);
    this->me.allMessage("offset = %9ld length = %9ld\n", this->offset, this->length);
    this->me.rootMessage("\n");
}
