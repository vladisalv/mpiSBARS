#ifndef __MATRIX_MPI_HEADER__
#define __MATRIX_MPI_HEADER__

#include "myMPI.h"
#include "dataMPI.h"

#include <fstream>

template<class DataType, class LengthData>
class MatrixMPI : public DataMPI<DataType, LengthData> {
protected:
    LengthData width, height;
    LengthData offset_row, offset_column;

    virtual void readMPI(char *file_name);
    virtual void readUsually(char *file_name);
    virtual void readMy(char *file_name);

    virtual void writeMPI(char *file_name);
    virtual void writeUsually(char *file_name);
    virtual void writeMy(char *file_name);
public:
    MatrixMPI(MyMPI me, const char *class_name, MPI_Datatype MpiDataType);
    ~MatrixMPI();
    virtual void debugInfo(const char *file, int line, const char *info = 0);
};

#include "matrixMPI.tcc"

#endif /* __MATRIX_MPI_HEADER__ */
