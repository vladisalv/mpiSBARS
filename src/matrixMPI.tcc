#include "matrixMPI.h"

template <class DataType, class LengthData>
MatrixMPI<DataType, LengthData>::MatrixMPI(MyMPI me, const char *class_name, MPI_Datatype MpiDataType)
    : DataMPI<DataType, LengthData>(me, class_name, MpiDataType), width(0), height(0), offset_row(0), offset_column(0)
{
}

template <class DataType, class LengthData>
MatrixMPI<DataType, LengthData>::~MatrixMPI()
{
}


template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::readMPI(char *file_name)
{
#ifdef USE_MPI
#endif
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
#ifdef USE_MPI
    LengthData *offset, *sum_offset;
    ulong common_height, common_width, common_length;
    common_width = width;
    common_height = this->offsetLength(offset, sum_offset, &this->height);
    common_length = common_width * common_height;

    MPI_File hFile;
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &hFile);
    MPI_Offset offsetHead = (this->me.isRoot() ? 0 : 3);
    MPI_Offset offsetFile = offsetHead + width * sum_offset[this->me.getRank()];
    MPI_File_set_view(hFile, offsetFile, this->MpiDataType, this->MpiDataType,
                      (char *)"native", MPI_INFO_NULL);

    if (this->me.isRoot()) {
        MPI_File_write(hFile, &common_height, 1, MPI_DOUBLE, 0);
        MPI_File_write(hFile, &common_width,  1, MPI_DOUBLE, 0);
        MPI_File_write(hFile, &common_length, 1, MPI_DOUBLE, 0);
    }

    MPI_File_write(hFile, this->data, this->length, MPI_DOUBLE, 0);
    MPI_File_close(&hFile);
#endif
}

template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::writeUsually(char *file_name)
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

template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::writeMy(char *file_name)
{
    ;
}


template <class DataType, class LengthData>
void MatrixMPI<DataType, LengthData>::debugInfo(const char *file, int line, const char *info)
{
    this->me.rootMessage("\n");
    this->me.rootMessage("This is debugInfo(%s) of %s in %s at line %d\n", info, this->class_name, file, line);
    this->me.allMessage("offset_row = %9ld height = %9ld offset_column = %9ld width = %9ld\n",
            this->offset_row, this->height, this->offset_column, this->width);
    this->me.rootMessage("\n");
}
