#ifndef __MY_MPI_HEADER__
#define __MY_MPI_HEADER__

// @TODO: time_measure in myMPI and myMPI in time_measure
//class TimeMeasure;

#include <mpi.h>

#include "support.h"
#include "types.h"

#include <stdarg.h>

#define NUMBER_OF_ROOT 0

class TimeMeasure;

class MyMPI {
    static int numberObject;
    char rank_str[10];
    TimeMeasure *mpi_time;
protected:
    int rank, size;
    int root_is, first_is, last_is;
    bool root, first, last, single;
public:
    MPI_Comm comm;

    MyMPI(MPI_Comm comm = MPI_COMM_WORLD, int argc = 0, char *argv[] = 0);
    MyMPI(const MyMPI &other);
    ~MyMPI();

    void iSend(void *buf, uint count, MPI_Datatype type, 
                int dest, int tag, MPI_Request *request);
    void iRecv(void *buf, uint count, MPI_Datatype type,
                int sourse, int tag, MPI_Request *request);
    void Send(void *buf, uint count, MPI_Datatype type, int dest, int tag);
    void Recv(void *buf, uint count, MPI_Datatype type, int dest, int tag);
    void wait(MPI_Request *req, MPI_Status *status);
    bool iProbe(int source, int tag);

    // usually function
    void Bcast(void *buf, uint len, MPI_Datatype type);
    void Synchronize();
    bool Test(MPI_Request *request);
    bool Probe(int source, int tag);

    void Gather(void *send_buf, uint send_len, MPI_Datatype send_type,
                void *recv_buf, uint recv_len, MPI_Datatype recv_type);
    void Allgather(void *send_buf, uint send_len, MPI_Datatype send_type,
                  void *recv_buf, uint recv_len, MPI_Datatype recv_type);

    MPI_File openFile(char *filename, int amode, MPI_Info info);
    MPI_Offset getSizeFile(MPI_File fh);
    void closeFile(MPI_File *fh);
    void readFile(MPI_File fh, MPI_Offset offset, void *mas, ulong length,
                  MPI_Datatype type, MPI_Info info);
    void writeFile(MPI_File fh, MPI_Offset offset, void *mas, ulong length,
                  MPI_Datatype type, MPI_Info info);

    ulong sumLength(ulong *length, ulong *last_length);

    int getRank();
    int getSize();
    double getTime();
    bool isRoot();
    bool isFirst();
    bool isLast();
    bool isSingle();
    int whoRoot();
    int whoFirst();
    int whoLast();
    char *getStrRank();

    void debugInfo(const char *str = 0);

    void rootMessage(const char *format, ...);
    void allMessage(const char *format, ...);
    void myMessage(int myrank, const char *format, ...);
};

#endif /* __MY_MPI_HEADER__ */
