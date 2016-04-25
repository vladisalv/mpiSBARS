#include "matrixMPI.h"

template <class DataType>
MatrixMPI<DataType>::MatrixMPI(MyMPI me)
    : DataMPI<DataType>(me), width(0), height(0), offset_row(0), offset_column(0)
{
}

template <class DataType>
MatrixMPI<DataType>::~MatrixMPI()
{
}


template <class DataType>
void MatrixMPI<DataType>::readMPI(char *file_name)
{
#ifdef USE_MPI
#endif
}

template <class DataType>
void MatrixMPI<DataType>::readUsually(char *file_name)
{
    ;
}

template <class DataType>
void MatrixMPI<DataType>::readMy(char *file_name)
{
    ;
}


template <class DataType>
void MatrixMPI<DataType>::writeMPI(char *file_name)
{
#ifdef USE_MPI
    ulong *offset, *sum_offset;
    ulong common_height, common_width, common_length;
    common_width = width;
    common_height = this->offsetLength(offset, sum_offset, &this->height);
    common_length = common_width * common_height;

    MPI_File hFile;
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &hFile);
    MPI_Offset offsetHead = (this->me.isRoot() ? 0 : 3);
    MPI_Offset offsetFile = offsetHead + width * sum_offset[this->me.getRank()];
    MPI_File_set_view(hFile, offsetFile, this->getMpiDataType(), this->getMpiDataType(),
                      (char *)"native", MPI_INFO_NULL);

    if (this->me.isRoot()) {
        MPI_File_write(hFile, &common_height, 1, MPI_DOUBLE, 0);
        MPI_File_write(hFile, &common_width,  1, MPI_DOUBLE, 0);
        MPI_File_write(hFile, &common_length, 1, MPI_DOUBLE, 0);
    }

    MPI_File_write(hFile, this->data, this->length, this->getMpiDataType(), 0);
    MPI_File_close(&hFile);
#endif
}

template <class DataType>
void MatrixMPI<DataType>::writeUsually(char *file_name)
{
    for (int i = 0; i < this->me.getSize(); i++) {
        this->me.Synchronize();
        if (this->me.getRank() == i) {
            FILE *file = fopen(file_name, "a");
            for (ulong j = 0; j < this->length; j++)
                ;//fprintf(file, "%.0f\n", this->data[j]);
            fflush(file);
            fclose(file);
        }
        this->me.Synchronize();
    }
}

template <class DataType>
void MatrixMPI<DataType>::writeMy(char *file_name)
{
    ;
}


template <class DataType>
void MatrixMPI<DataType>::debugInfo(const char *file, int line, const char *info)
{
    this->me.rootMessage("\n");
    this->me.rootMessage("This is debugInfo(%s) in %s at line %d\n", info, file, line);
    this->me.allMessage("offset_row = %9ld height = %9ld offset_column = %9ld width = %9ld\n",
            this->offset_row, this->height, this->offset_column, this->width);
    this->me.rootMessage("\n");
}
