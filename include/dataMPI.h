#ifndef __DATA_MPI_HEADER__
#define __DATA_MPI_HEADER__

#include "myMPI.h"

template<class DataType, class LengthData>
class DataMPI {
protected:
    MyMPI me;
    DataType *data;
    LengthData length;

    virtual void readMPI(char *file_name) = 0;
    virtual void readUsually(char *file_name) = 0;
    virtual void readMy(char *file_name) = 0;

    virtual void writeMPI(char *file_name) = 0;
    virtual void writeUsually(char *file_name) = 0;
    virtual void writeMy(char *file_name) = 0;

    LengthData offsetLength(LengthData* &offset, LengthData* &sum_offset, LengthData *var);
public:
    DataMPI(MyMPI new_me);
    virtual ~DataMPI();

    bool isEmpty();

    virtual void readFile(char *file_name);
    virtual void writeFile(char *file_name);
    void free();
};

#include "dataMPI.tcc"

#endif /* __DATA_MPI_HEADER__ */
