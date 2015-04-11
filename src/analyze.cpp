#include "analyze.h"

Analyze::Analyze(MyMPI me_new, GpuComputing gpu_new, double eps_new, ulong min_length_new, double fidelity_repeat_new, size_t limit_memory_new)
    : me(me_new), gpu(gpu_new),  eps(eps_new), min_length(min_length_new), fidelity_repeat(fidelity_repeat_new), limit_memory(limit_memory_new), dec_other(0), buf_tmp(0)
{
}

Analyze::~Analyze()
{
}


SetRepeats Analyze::doAnalyze(MatrixGomology matrixGomology)
{
    SetRepeats setRepeats(me);
    long x_end, y_end, length;
    setRepeats.x_limit_left = matrixGomology.offset_column;
    setRepeats.x_limit_right = matrixGomology.offset_column + matrixGomology.width - 1;
    setRepeats.y_limit_above = matrixGomology.offset_row;
    setRepeats.y_limit_bottom = matrixGomology.offset_row + matrixGomology.height - 1;
    for (ulong y = 0; y < matrixGomology.height; y++)
        for (ulong x = 0; x < matrixGomology.width; x++)
            if (matrixGomology.data[y * matrixGomology.width + x]) {
                x_end = x;
                y_end = y;
                length = 0;
                while (x_end < matrixGomology.width && y_end < matrixGomology.height
                    && matrixGomology.data[y_end * matrixGomology.width + x_end]) {
                    matrixGomology.data[y_end * matrixGomology.width + x_end] = false;
                    x_end++;
                    y_end++;
                    length++;
                }
                Repeat rep(x + matrixGomology.offset_column, y + matrixGomology.offset_row,
                        --x_end + matrixGomology.offset_column, --y_end + matrixGomology.offset_row, length);
                setRepeats.vec.push_back(rep);
            }
    return setRepeats;
}

void Analyze::recvDecompositon()
{
    while (!recvDecompositonAsync())
        ;
    waitDecomposition();
}

bool Analyze::recvDecompositonAsync()
{
    int flag = 1;
    if (!buf_tmp) {
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int source_proc, size_dec_other;
            source_proc = status.MPI_SOURCE;
            MPI_Get_count(&status, MPI_TFLOAT, &size_dec_other);
            buf_tmp = new TypeDecomposition [size_dec_other];
            me.iRecv(buf_tmp, size_dec_other, MPI_TFLOAT, source_proc, 1, &req);
        }
    }
    return flag;
}

void Analyze::waitDecomposition()
{
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    delete [] dec_other;
    dec_other = buf_tmp;
    buf_tmp = 0;
}


SetRepeats Analyze::doAnalyze(Decomposition decomposition)
{
    MPI_Request *req_send = new MPI_Request [me.getSize()];
    for (int i = 0; i < me.getSize(); i++)
        me.iSend(decomposition.data, decomposition.length, MPI_TFLOAT, i, 1, &req_send[i]);

    for (int i = 0; i < me.getSize(); i++) {
        recvDecompositon();
    }

    SetRepeats result(me);
    ulong size = sqrt(limit_memory);
    return result;
}

SetRepeats Analyze::doAnalyze(Decomposition decomposition1, Decomposition decomposition2)
{
    SetRepeats result(me);
    return result;
}

SetRepeats Analyze::comparisonRepeats(SetRepeats setRepeats1, SetRepeats setRepeats2)
{
    SetRepeats result(me);
    return result;
}


double Analyze::getEps()
{
    return eps;
}

ulong  Analyze::getMinLengthRepeat()
{
    return min_length;
}

double Analyze::getFidelityRepeat()
{
    return fidelity_repeat;
}

size_t Analyze::getLimitMemoryMatrix()
{
    return limit_memory;
}


void Analyze::setEps(double eps_new)
{
    eps = eps_new ;
}

void Analyze::setMinLengthRepeat(ulong min_length_new)
{
    min_length = min_length_new;
}

void Analyze::setFidelityRepeat(double fidelity_repeat_new)
{
    fidelity_repeat = fidelity_repeat_new;
}

void Analyze::setLimitMemoryMatrix(size_t limit_memory_new)
{
    limit_memory = limit_memory_new;
}
