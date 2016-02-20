#ifdef USE_MPI

#include "myMPI.h"
#include "time_measure.h"
#include "support.h"

int MyMPI::numberObject = 0;

MyMPI::MyMPI(MPI_Comm new_comm, int argc, char *argv[])
{
    if (numberObject == 0) {
        int init;
        MPI_Initialized(&init);
        if (!init)
            MPI_Init(&argc, &argv);
    }

    comm = new_comm;
    numberObject++;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    root_is = NUMBER_OF_ROOT;
    first_is = 0;
    last_is = size - 1;

    root   = (rank == root_is);
    first  = (rank == first_is);
    last   = (rank == last_is);
    single = (size == 1);

    strcpy(rank_str, rank_to_string(rank, size));

    mpi_time = new TimeMeasure(this);
}

MyMPI::MyMPI(const MyMPI& other)
{
    comm = other.comm;
    numberObject++;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    root_is = NUMBER_OF_ROOT;
    first_is = 0;
    last_is = size - 1;

    root   = (rank == root_is);
    first  = (rank == first_is);
    last   = (rank == last_is);
    single = (size == 1);

    strcpy(rank_str, rank_to_string(rank, size));

    mpi_time = new TimeMeasure(this);
}

MyMPI::~MyMPI()
{
    numberObject--;
    if (numberObject == 0) {
        Synchronize();
        MPI_Finalize();
    }
}

void MyMPI::iSend(void *buf, uint count, MPI_Datatype type,
                    int dest, int tag, MPI_Request *request)
{
    MPI_Isend(buf, count, type, dest, tag, comm, request);
}

void MyMPI::iRecv(void *buf, uint count, MPI_Datatype type,
                    int sourse, int tag, MPI_Request *request)
{
    MPI_Irecv(buf, count, type, sourse, tag, comm, request);
}

void MyMPI::Send(void *buf, uint count, MPI_Datatype type, int dest, int tag)
{
    MPI_Send(buf, count, type, dest, tag, comm);
}

void MyMPI::Recv(void *buf, uint count, MPI_Datatype type, int dest, int tag)
{
    MPI_Recv(buf, count, type, dest, tag, comm, MPI_STATUS_IGNORE);
}


void MyMPI::wait(MPI_Request *request, MPI_Status *status)
{
    MPI_Wait(request, status);
}

bool MyMPI::iProbe(int source, int tag)
{
    int *flag;
    MPI_Status status;
    MPI_Iprobe(source, tag, comm, flag, &status);
    if (*flag == 0)
        return false;
    else
        return true;
}


void MyMPI::Bcast(void *buf, uint len, MPI_Datatype type)
{
    MPI_Bcast(buf, len, type, root_is, comm);
}

void MyMPI::Synchronize()
{
    MPI_Barrier(comm);
}

bool MyMPI::Test(MPI_Request *request)
{
    int flag = 0;
    MPI_Test(request, &flag, MPI_STATUS_IGNORE);
    return flag;
}

bool MyMPI::Probe(int source, int tag)
{
    int flag;
    MPI_Iprobe(source, tag, comm, &flag, MPI_STATUS_IGNORE);
    return flag;
}

void MyMPI::Gather(void *send_buf, uint send_len, MPI_Datatype send_type,
                   void *recv_buf, uint recv_len, MPI_Datatype recv_type)
{
    MPI_Gather(send_buf, send_len, send_type, recv_buf, recv_len, recv_type, root_is, comm);
}

void MyMPI::Allgather(void *send_buf, uint send_len, MPI_Datatype send_type,
                      void *recv_buf, uint recv_len, MPI_Datatype recv_type)
{
    MPI_Allgather(send_buf, send_len, send_type, recv_buf, recv_len, recv_type, comm);
}

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


ulong MyMPI::sumLength(ulong *length, ulong *last_length)
{
    ulong *all_length, sum, last_elem;
    sum = 0;
    all_length = new ulong [size];
    // @TODO: may be only create all_length with new in root?
    Gather(length, 1, MPI_LONG, all_length, 1, MPI_LONG);
    if (root) {
        for (int i = 0; i < size; i++)
            sum += all_length[i];
        last_elem = all_length[last_is];
    }
    Bcast(&sum, 1, MPI_LONG);
    Bcast(&last_elem, 1, MPI_LONG);
    *last_length = last_elem;
    return sum;
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

void MyMPI::debugInfo(const char *file, int line, const char *info)
{
    fflush(stdout);
    if (root) {
        printf("\n");
        printf("This is debugInfo(%s) of %s in %s at line %d\n", info, "MyMPI", file, line);
        printf("We have %d process. I'm rank = %d\n", size, rank);
        printf("Now we have in system %d MPI object.\n", numberObject);
        if (comm == MPI_COMM_WORLD)
            printf("Your comm = MPI_COMM_WORLD\n");
        else
            printf("Your comm NOT MPI_COMM_WORLD\n");
        printf("\n");
    }
    fflush(stdout);
}

void MyMPI::rootMessage(const char *format, ...)
{
    fflush(stdout);
    Synchronize();
    fflush(stdout);
    if (root) {
        va_list arg;
        va_start(arg, format);
        vprintf(format, arg);
        va_end(arg);
    }
    fflush(stdout);
    Synchronize();
    fflush(stdout);
}

void MyMPI::allMessage(const char *format, ...)
{
    fflush(stdout);
    Synchronize();
    fflush(stdout);
    for (int i = 0; i < size; i++) {
        fflush(stdout);
        if (i == rank) {
            //printf("[%s]:", rank_str);
            va_list arg;
            va_start(arg, format);
            vprintf(format, arg);
            va_end(arg);
            fflush(stdout);
        }
        fflush(stdout);
        Synchronize();
        fflush(stdout);
    }
    //if (rank == root)
        //printf("\n\n");
    fflush(stdout);
    Synchronize();
    fflush(stdout);
}

void MyMPI::myMessage(int myrank, const char *format, ...)
{
    fflush(stdout);
    if (rank == myrank) {
        printf("Process %s: ", rank_str);
        va_list arg;
        va_start(arg, format);
        vprintf(format, arg);
        va_end(arg);
    }
    fflush(stdout);
}

#endif /* USE_MPI */
