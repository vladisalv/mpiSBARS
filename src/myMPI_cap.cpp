#ifndef USE_MPI

#include "myMPI.h"
#include "time_measure.h"

int MyMPI::numberObject = 0;

MyMPI::MyMPI(int argc, char *argv[])
{
}

MyMPI::MyMPI(const MyMPI& other)
{
}

MyMPI::~MyMPI()
{
}

void MyMPI::iSend(void *buf, uint count, MPI_Datatype type,
                    int dest, int tag, MPI_Request *request)
{
}

void MyMPI::iRecv(void *buf, uint count, MPI_Datatype type,
                    int sourse, int tag, MPI_Request *request)
{
}

void MyMPI::Send(void *buf, uint count, MPI_Datatype type, int dest, int tag)
{
}

void MyMPI::Recv(void *buf, uint count, MPI_Datatype type, int dest, int tag)
{
}


void MyMPI::wait(MPI_Request *request, MPI_Status *status)
{
}

bool MyMPI::iProbe(int source, int tag)
{
    return true;
}


void MyMPI::Bcast(void *buf, uint len, MPI_Datatype type)
{
}

void MyMPI::Synchronize()
{
}

bool MyMPI::Test(MPI_Request *request)
{
    return true;
}

bool MyMPI::Probe(int source, int tag)
{
    return true;
}

void MyMPI::Gather(void *send_buf, uint send_len, MPI_Datatype send_type,
                   void *recv_buf, uint recv_len, MPI_Datatype recv_type)
{
}

void MyMPI::Allgather(void *send_buf, uint send_len, MPI_Datatype send_type,
                      void *recv_buf, uint recv_len, MPI_Datatype recv_type)
{
}

/*
MPI_File MyMPI::openFile(char *filename, int amode, MPI_Info info)
{
    MPI_File fh;
    MPI_File_open(comm, filename, amode, info, &fh);
    return fh;
}

MPI_Offset MyMPI::getSizeFile(MPI_File fh)
{
    MPI_Offset offset;
    MPI_File_get_size(fh, &offset);
    return offset;
}

void MyMPI::closeFile(MPI_File *fh)
{
    MPI_File_close(fh);
}


void MyMPI::readFile(MPI_File fh, MPI_Offset offset, void *mas, ulong length,
              MPI_Datatype type, MPI_Info info)
{
    MPI_File_set_view(fh, 0, type, type, (char *)"native", info);
    MPI_File_read_at_all(fh, offset, mas, length, type, 0);
}

void MyMPI::writeFile(MPI_File fh, MPI_Offset offset, void *mas, ulong length,
              MPI_Datatype type, MPI_Info info)
{
    MPI_File_set_view(fh, 0, type, type, (char *)"native", info);
    MPI_File_write_at_all(fh, offset, mas, length, type, 0);
}
*/


ulong MyMPI::sumLength(ulong *length, ulong *last_length)
{
    return 0;
}


int MyMPI::getRank()
{
    return rank;
}

int MyMPI::getSize()
{
    return size;
}

double MyMPI::getTime()
{
    return MPI_Wtime();
}

bool MyMPI::isRoot()
{
    return root;
}

bool MyMPI::isFirst()
{
    return first;
}

bool MyMPI::isLast()
{
    return last;
}

bool MyMPI::isSingle()
{
    return single;
}

int MyMPI::whoRoot()
{
    return root_is;
}

int MyMPI::whoFirst()
{
    return first_is;
}

int MyMPI::whoLast()
{
    return last_is;
}

char *MyMPI::getStrRank()
{
    return rank_str;
}

void MyMPI::debugInfo(const char *str)
{
    if (root) {
        printf("\n");
        printf("Hi! I'm root. This is debugInfo(%s) of MyMPI\n", str);
        printf("We have %d process. I'm rank = %d\n", size, rank);
        printf("Now we have in system %d MPI object.\n", numberObject);
        printf("\n");
        fflush(stdout);
    }
}

void MyMPI::rootMessage(const char *format, ...)
{
    if (root) {
        va_list arg;
        va_start(arg, format);
        vprintf(format, arg);
        va_end(arg);
        fflush(stdout);
    }
}

void MyMPI::allMessage(const char *format, ...)
{
    fflush(stdout);
    Synchronize();
    if (rank == root)
        printf("\n");
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            printf("[%s]:\n", rank_str);
            va_list arg;
            va_start(arg, format);
            vprintf(format, arg);
            va_end(arg);
            fflush(stdout);
        }
        Synchronize();
    }
    if (rank == root)
        printf("\n\n");
    Synchronize();
    fflush(stdout);
}

void MyMPI::myMessage(int myrank, const char *format, ...)
{
    if (rank == myrank) {
        printf("Process %s: ", rank_str);
        va_list arg;
        va_start(arg, format);
        vprintf(format, arg);
        va_end(arg);
        fflush(stdout);
    }
}

#endif /* USE_MPI */
