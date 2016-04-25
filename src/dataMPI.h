#ifndef __DATA_MPI_HEADER__
#define __DATA_MPI_HEADER__

#include "types.h"
#include "myMPI.h"

template<class DataType>
class DataMPI {
protected:
    MyMPI me;
    DataType *data;
    ulong length;

    virtual void readMPI(char *file_name) = 0;
    virtual void readUsually(char *file_name) = 0;
    virtual void readMy(char *file_name) = 0;

    virtual void writeMPI(char *file_name) = 0;
    virtual void writeUsually(char *file_name) = 0;
    virtual void writeMy(char *file_name) = 0;

    ulong offsetLength(ulong* &offset, ulong* &sum_offset, ulong *var);
public:
    DataMPI(MyMPI new_me);
    virtual ~DataMPI();

    bool isEmpty();

    virtual void readFile(char *file_name);
    virtual void writeFile(char *file_name);
    virtual void debugInfo(const char *file, int line, const char *info = 0) = 0;
    MPI_Datatype getMpiDataType();
    void free();
    typedef DataType data_type;
};

#include "dataMPI.tcc"

#endif /* __DATA_MPI_HEADER__ */
