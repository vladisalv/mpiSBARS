#include "compare.h"

Compare::Compare(MyMPI new_me, GpuComputing new_gpu, double eps_new)
    : me(new_me), gpu(new_gpu), eps(eps_new)
{
}

Compare::~Compare()
{
}

MatrixGomology Compare::doCompare(Decomposition &decompose)
{
    return compareSelf(decompose);
}

MatrixGomology Compare::doCompare(Decomposition &decompose1, Decomposition &decompose2)
{
    return compareTwo(decompose1, decompose2);
}

MatrixGomology Compare::comparisonMatrix(MatrixGomology matrix1, MatrixGomology matrix2)
{
    for (int i = 0; i < matrix1.height; i++)
        for (int j = 0; j < matrix1.width; j++)
            if (!matrix2.data[i * matrix1.width + j])
                matrix1.data[i * matrix1.width + j] = false;
    MatrixGomology mat(me);
    mat.height = matrix1.height;
    mat.offset_row = matrix1.offset_row;
    mat.width = matrix1.width;
    mat.offset_column = matrix1.offset_column;
    mat.length = matrix1.length;
    mat.data = matrix1.data;
    return mat;
}

MatrixGomology Compare::compareSelf(Decomposition &decomposition)
{
    ulong *length_all, *sum_length_array;
    ulong *decompose_other_begin = new ulong [me.getSize()];
    ulong sum_all = decomposition.offsetLength(length_all, sum_length_array, &decomposition.height);
    for (int i = 0; i < me.getSize(); i++)
        decompose_other_begin[i] = sum_length_array[i] * decomposition.width;

    MPI_Request *req_send = new MPI_Request [me.getSize()];
    for (int i = 0; i < me.getSize(); i++)
        me.iSend(decomposition.data, decomposition.length, MPI_TFLOAT, i, 0, &req_send[i]);

    TypeDecomposition *decompose_other = new TypeDecomposition [sum_all * decomposition.width];
    MPI_Request *req_recv = new MPI_Request [me.getSize()];
    for (int i = 0; i < me.getSize(); i++) {
        me.iRecv(&decompose_other[decompose_other_begin[i]],
                 decomposition.width * length_all[i], MPI_TFLOAT, i, 0, &req_recv[i]);
    }

    MatrixGomology matrixGomology(me);
    matrixGomology.length = decomposition.height * sum_all;
    matrixGomology.height = decomposition.height;
    matrixGomology.offset_row = decomposition.offset_row;
    matrixGomology.width = sum_all;
    matrixGomology.offset_column = 0;
    matrixGomology.data = new bool [matrixGomology.length];
    for (uint i = 0; i < matrixGomology.length; i++)
        matrixGomology.data[i] = false;

    int num_send = 0;
    bool *send_flag = new bool [me.getSize()];
    for (int i = 0; i < me.getSize(); i++)
        send_flag[i] = false;
    while (num_send < me.getSize()) {
        for (int i = 0; i < me.getSize(); i++) {
            if (!send_flag[i] && me.Test(&req_recv[i])) {
                compareDecomposition(decomposition.data, decomposition.height,
                                    &decompose_other[decompose_other_begin[i]],
                                    length_all[i],
                                    decomposition.width, matrixGomology.data,
                                    sum_length_array[i], sum_all);
                send_flag[i] = true;
                num_send++;
            }
        }
    }
    delete [] length_all;
    delete [] sum_length_array;
    delete [] decompose_other_begin;
    delete [] decompose_other;
    delete [] req_recv;
    delete [] send_flag;
    return matrixGomology;
}

MatrixGomology Compare::compareTwo(Decomposition &decomposition1, Decomposition &decomposition2)
{
    // TEST!!! above use offsetLength
    ulong length_all[me.getSize()];
    //ulong *length = new ulong [me.getSize()]; ????
    me.Allgather(&decomposition2.height, 1, MPI_UNSIGNED_LONG,
                 &length_all,            1, MPI_UNSIGNED_LONG);

    MPI_Request *req_send = new MPI_Request [me.getSize()];
    for (int i = 0; i < me.getSize(); i++)
        me.iSend(decomposition2.data, decomposition2.length, MPI_TFLOAT, i, 0, &req_send[i]);

    ulong sum_all = 0;
    ulong *decompose_other_begin = new ulong [me.getSize()];
    ulong *sum_length_array  =  new ulong [me.getSize()];
    for (int i = 0; i < me.getSize(); i++) {
        decompose_other_begin[i] = sum_all * decomposition2.width;
        sum_length_array[i] = sum_all;
        sum_all += length_all[i];
    }

    TypeDecomposition *decompose_other = new TypeDecomposition [sum_all * decomposition2.width];
    MPI_Request *req_recv = new MPI_Request [me.getSize()];
    for (int i = 0; i < me.getSize(); i++) {
        me.iRecv(&decompose_other[decompose_other_begin[i]],
                 decomposition2.width * length_all[i], MPI_TFLOAT, i, 0, &req_recv[i]);
    }

    MatrixGomology matrixGomology(me);
    matrixGomology.length = decomposition1.height * sum_all;
    matrixGomology.height = decomposition1.height;
    matrixGomology.offset_row = decomposition1.offset_row;
    matrixGomology.width = sum_all;
    matrixGomology.offset_column = 0;
    matrixGomology.data = new bool [matrixGomology.length];
    for (uint i = 0; i < matrixGomology.length; i++)
        matrixGomology.data[i] = false;

    int num_send = 0;
    bool *send_flag = new bool [me.getSize()];
    for (int i = 0; i < me.getSize(); i++)
        send_flag[i] = false;
    while (num_send < me.getSize()) {
        for (int i = 0; i < me.getSize(); i++) {
            if (!send_flag[i] && me.Test(&req_recv[i])) {
                compareDecomposition(decomposition1.data, decomposition1.height,
                                    &decompose_other[decompose_other_begin[i]],
                                    length_all[i],
                                    decomposition1.width, matrixGomology.data,
                                    sum_length_array[i], sum_all);
                send_flag[i] = true;
                num_send++;
            }
        }
    }
    return matrixGomology;
}

void Compare::compareDecomposition(TypeDecomposition *decompose1, ulong length_decompose1,
                                   TypeDecomposition *decompose2, ulong length_decompose2,
                                   ulong width, TypeGomology *data, ulong begin, ulong sum_all)
{
    if (gpu.isUse())
        gpu.compareDecompositionGpu2(decompose1, length_decompose1, decompose2, length_decompose2, width, data, begin, sum_all, eps);
    else
        compareDecompositionHost(decompose1, length_decompose1, decompose2, length_decompose2, width, data, begin, sum_all);
}

void Compare::compareDecompositionHost(TypeDecomposition *decompose1, ulong length_decompose1,
                                       TypeDecomposition *decompose2, ulong length_decompose2,
                                       ulong width, TypeGomology *data, ulong begin, ulong sum_all)
{
    bool answer;
    for (int i = 0; i < length_decompose1; i++)
        for (int j = 0; j < length_decompose2; j++) {
            answer = compareVector(&decompose1[i * width], &decompose2[j * width], width);
            data[i * sum_all + begin + j] = answer;
        }
}

bool Compare::compareVector(TypeDecomposition *vec1, TypeDecomposition *vec2, ulong length)
{
    TypeDecomposition sum = 0;
    double eps_2 = eps;
    for (int i = 0; i < length; i++) {
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
        if (sum > eps_2)
            break;
    }
    if (sum > eps_2)
        return false;
    else
        return true;
}

double Compare::getEps()
{
    return eps;
}

void Compare::setEps(double eps_new)
{
    eps = eps_new;
}
