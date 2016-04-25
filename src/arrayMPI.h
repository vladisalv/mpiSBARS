#ifndef __ARRAY_MPI_HEADER__
#define __ARRAY_MPI_HEADER__

#include "types.h"
#include "myMPI.h"
#include "dataMPI.h"

template <class DataType>
class ArrayMPI : public DataMPI<DataType> {
protected:
    ulong offset;

    virtual void readMPI(char *file_name);
    virtual void readUsually(char *file_name);
    virtual void readMy(char *file_name);

    virtual void writeMPI(char *file_name);
    virtual void writeUsually(char *file_name);
    virtual void writeMy(char *file_name);
public:
    ArrayMPI(MyMPI me);
    virtual ~ArrayMPI();
    virtual void debugInfo(const char *file, int line, const char *info = 0);
};

#include "arrayMPI.tcc"

#endif /* __ARRAY_MPI_HEADER__ */
