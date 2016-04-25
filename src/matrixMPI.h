#ifndef __MATRIX_MPI_HEADER__
#define __MATRIX_MPI_HEADER__

#include "types.h"
#include "myMPI.h"
#include "dataMPI.h"

template<class DataType>
class MatrixMPI : public DataMPI<DataType> {
protected:
    ulong width, height;
    ulong offset_row, offset_column;

    virtual void readMPI(char *file_name);
    virtual void readUsually(char *file_name);
    virtual void readMy(char *file_name);

    virtual void writeMPI(char *file_name);
    virtual void writeUsually(char *file_name);
    virtual void writeMy(char *file_name);
public:
    MatrixMPI(MyMPI me);
    ~MatrixMPI();
    virtual void debugInfo(const char *file, int line, const char *info = 0);
};

#include "matrixMPI.tcc"

#endif /* __MATRIX_MPI_HEADER__ */
