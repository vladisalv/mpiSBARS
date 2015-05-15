#include "analyze.h"

Analyze::Analyze(MyMPI me_new, GpuComputing gpu_new, double eps_new, ulong min_length_new, double fidelity_repeat_new, size_t limit_memory_new)
    : me(me_new), gpu(gpu_new),  eps(eps_new), min_length(min_length_new),
        fidelity_repeat(fidelity_repeat_new), limit_memory(limit_memory_new),
        decomposition(me), dec_other(me), buf_tmp(0), source_proc(0)
{
}

Analyze::~Analyze()
{
}


ListRepeats Analyze::doAnalyze(MatrixGomology matrixGomology)
{
    ListRepeats listRepeats(me);
    long x_end, y_end, length;
    listRepeats.x_limit_left   = matrixGomology.offset_column;
    listRepeats.x_limit_right  = matrixGomology.offset_column + matrixGomology.width - 1;
    listRepeats.y_limit_above  = matrixGomology.offset_row;
    listRepeats.y_limit_bottom = matrixGomology.offset_row + matrixGomology.height - 1;
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
                listRepeats.data->push_back(rep);
            }
    return listRepeats;
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
            int size_dec_other;
            source_proc = status.MPI_SOURCE;
            MPI_Get_count(&status, MPI_TFLOAT, &size_dec_other);
            buf_tmp = new TypeDecomposition [size_dec_other];
            me.iRecv(buf_tmp, size_dec_other, MPI_TFLOAT, source_proc, 1, &req);
            dec_other.length = size_dec_other;
            dec_other.width  = decomposition.width;
            dec_other.height = dec_other.length / dec_other.width;
        }
    }
    return flag;
}

void Analyze::waitDecomposition()
{
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    dec_other.free();
    dec_other.data = buf_tmp;
    buf_tmp = 0;
}


ListRepeats Analyze::doAnalyze(Decomposition myDecomposition)
{
    decomposition = myDecomposition;

    MPI_Request *req_send = new MPI_Request [me.getSize()];
    for (int i = 0; i < me.getSize(); i++)
        me.iSend(decomposition.data, decomposition.length, MPI_TFLOAT, i, 1, &req_send[i]);

    ListRepeats *resultProc = static_cast<ListRepeats*>(operator new[](me.getSize() * sizeof(ListRepeats)));
    for (int i = 0; i < me.getSize(); i++) {
        new (resultProc + i) ListRepeats(me);
    }

    MatrixGomology matrixGomology(me);
    ulong height_block, width_block;
    height_block = width_block = sqrt(limit_memory);
    size_t size_block = height_block * width_block * sizeof(TypeGomology);
    matrixGomology.data = (TypeGomology *)malloc(size_block);
    memset(matrixGomology.data, 0, size_block);
    Compare compare(me, gpu, eps);
    ulong *height_other = new ulong [me.getSize()];
    for (int proc = 0; proc < me.getSize(); proc++) {
        recvDecompositon();
        if (me.isSingle()) { // i do not know why
            dec_other.free();
            dec_other.data = new TypeDecomposition [decomposition.length];
            dec_other.length = decomposition.length;
            dec_other.height = decomposition.height;
            dec_other.width  = decomposition.width;
            memcpy(dec_other.data, decomposition.data, dec_other.length * sizeof(TypeDecomposition));
        }
        height_other[source_proc] = dec_other.height;
        for (int j = 0; j < (dec_other.height + width_block - 1) / width_block; j++) {
            ListRepeats resultColumn(me);
            for (int i = 0; i < (decomposition.height + height_block - 1) / height_block; i++) {
                ListRepeats resultBlock(me);
                ulong height_block_now = (i == decomposition.height / height_block) ? decomposition.height % height_block : height_block;
                ulong width_block_now  = (j == dec_other.height     / width_block ) ? dec_other.height     % width_block  : width_block;
                memset(matrixGomology.data, 0, size_block);
                compare.compareDecomposition(
                                    &decomposition.data[(i * height_block) * decomposition.width],
                                    height_block_now,
                                    &dec_other.data[(j * width_block) * decomposition.width],
                                    width_block_now,
                                    decomposition.width,
                                    matrixGomology.data,
                                    0, width_block_now
                                    );
                matrixGomology.height = height_block_now;
                matrixGomology.width  = width_block_now;
                matrixGomology.offset_row    = height_block * i;
                matrixGomology.offset_column = width_block * j;
                resultBlock = doAnalyze(matrixGomology);
                //me.allMessage("%d %d %d %d %d %d %d \n", proc, i, j, height_block_now, width_block_now, resultBlock.y_limit_above, resultColumn.y_limit_bottom);
                //resultBlock.writeFile("t");
                resultColumn.mergeRepeatsRow(resultBlock);
            }
            //me.allMessage("%d %d resultColumn\n", proc, j);
            //resultColumn.writeFile("f");
            resultProc[source_proc].mergeRepeatsColumn(resultColumn);
        }
        //me.allMessage("%d resultProc\n", proc);
        //resultProc[proc].writeFile("f");
    }

    ulong *length_all, *sum_length_array;
    ulong sum_all = decomposition.offsetLength(length_all, sum_length_array, &decomposition.height);
    ulong myOffsetHeight = sum_length_array[me.getRank()];
    ulong myOffsetWidth  = 0;
    for (int i = 0; i < me.getSize(); i++) {
        //printf("^^^^^^^^^^^^i %d %d %d board: x_left=%ld x_right=%ld y_above=%ld y_bottom=%ld\n", me.getRank(), i, myOffsetHeight, resultProc[i].x_limit_left, resultProc[i].x_limit_right, resultProc[i].y_limit_above, resultProc[i].y_limit_bottom);
        resultProc[i].makeOffsetColumn(myOffsetWidth);
        resultProc[i].makeOffsetRow(myOffsetHeight);
        //printf("************i %d %d %d board: x_left=%ld x_right=%ld y_above=%ld y_bottom=%ld\n", me.getRank(), i, myOffsetHeight, resultProc[i].x_limit_left, resultProc[i].x_limit_right, resultProc[i].y_limit_above, resultProc[i].y_limit_bottom);
        myOffsetWidth += height_other[i];
    }

    ListRepeats result(me);
    for (int i = 0; i < me.getSize(); i++) {
        result.mergeRepeatsColumn(resultProc[i]);
        //printf("------------------------------i %d %d %d board: x_left=%ld x_right=%ld y_above=%ld y_bottom=%ld\n", me.getRank(), i, myOffsetHeight, result.x_limit_left, result.x_limit_right, result.y_limit_above, result.y_limit_bottom);
    }
    //printf("------------------------------i %d board: x_left=%ld x_right=%ld y_above=%ld y_bottom=%ld\n", me.getRank(), result.x_limit_left, result.x_limit_right, result.y_limit_above, result.y_limit_bottom);
    //result.writeFile("fds");

    for (int i = 0; i < me.getSize(); i++) {
        resultProc[i].~ListRepeats();
    }
    operator delete [] (resultProc);
    dec_other.free();
    delete [] height_other;

    return result;
}


/*
ListRepeats Analyze::doAnalyze(Decomposition myDecomposition)
{
    decomposition = myDecomposition;
    MPI_Request *req_send = new MPI_Request [me.getSize()];
    for (int i = 0; i < me.getSize(); i++)
        me.iSend(decomposition.data, decomposition.length, MPI_TFLOAT, i, 1, &req_send[i]);

#ifdef USE_CUDA
    TypeDecomposition *dec1dev, *dec2dev;
    TypeDecomposition *dec1host, *dec2host;
    dec1host = dec2host = decomposition.data;
    size_t size_decomposition = decomposition.length * sizeof(TypeDecomposition);
    HANDLE_ERROR(cudaMalloc((void **)&dec1dev, size_decomposition));
    HANDLE_ERROR(cudaMemcpy(dec1dev, dec1host, size_decomposition, cudaMemcpyHostToDevice));

    ulong height_block = sqrt(limit_memory) / 64;
    size_t size_block = 2 * height_block * height_block * sizeof(TypeDecomposition);
    TypeDecomposition *res1dev, *res2dev, *resHost;
    HANDLE_ERROR(cudaMalloc((void **)&res1dev, size_block));
    HANDLE_ERROR(cudaMalloc((void **)&res2dev, size_block));
    resHost = (TypeDecomposition *)malloc(size_block);
    //create stream
    cudaStream_t st1, st2;
    cudaStreamCreate(&st1);
    cudaStreamCreate(&st2);
    ListRepeats *resultProc = static_cast<ListRepeats*>(operator new[](me.getSize() * sizeof(ListRepeats)));
    for (int i = 0; i < me.getSize(); i++) {
        new (resultProc + i) ListRepeats(me);
    }
    for (int proc = 0; proc < me.getSize(); proc++) {
        recvDecompositon();
        ListRepeats resultColumn(me), resultBlock(me);
        bool flag = false;
        for (int j = 0; j < dec_other.height / (2 * size_block); j++) {
            // cudaStreamSynchronize(st2);
            // cudaMemcpyAsync st1
            // kenrnel run
            // analyze if (buf != 0)
            // merge - it can empty merge
            // compute last
            for (int i = 1; i < dec_other.height / size_block; i++) {
                // cudaStreamSynchronize(st1);
                // kenrnel run
                // cudaMemcpyAsync DtoH st2
                if (i == 1)
                    ;// merge last and resultColumn-resultBlock. it may be above
                else
                    ;// merge - it can empty merge
                // cudaStreamSynchronize(st2);
                // analyze if (buf != 0)
            }
            // cudaStreamSynchronize(st1);
            // cudaMemcpyAsync DtoH st2
            // merge - it can empty merge
        }
        // cudaStreamSynchronize(st2);
        // analyze if (buf != 0)
        // merge - it can empty merge
        // merge last and resultColumn-resultBlock. it may be above
        //resultProc[proc] = resultColumn;
    }
    ListRepeats result(me);
    // for result  *=  resultProc[i];
    for (int i = 0; i < me.getSize(); i++) {
        resultProc[i].~ListRepeats();
    }
    operator delete [] (resultProc);
    dec_other.free();
    cudaStreamDestroy(st1);
    cudaStreamDestroy(st2);
#endif

    return result;
}
*/

ListRepeats Analyze::doAnalyze(Decomposition decomposition1, Decomposition decomposition2)
{
    ListRepeats result(me);
    return result;
}

ListRepeats Analyze::comparisonRepeats(ListRepeats listRepeats1, ListRepeats listRepeats2)
{
    ListRepeats result(me);
    TypeAnalysis::iterator l1, l2;
    for (l1 = listRepeats1.data->begin(); l1 != listRepeats1.data->end(); l1++)
        for (l2 = listRepeats2.data->begin(); l2 != listRepeats2.data->end(); l2++)
            if (l1->y_begin - l1->x_begin == l2->y_begin - l2->x_begin) {
                if (l1->x_begin <= l2->x_begin && l1->x_end >= l2->x_end) {
                    result.data->push_back(*l2);
                } else if (l1->x_begin <= l2->x_begin && l1->x_end <= l2->x_end) {
                    Repeat tmp(l2->x_begin, l2->y_begin, l1->x_end, l1->y_end, l1->x_end - l2->x_begin + 1);
                    result.data->push_back(tmp);
                } else if (l2->x_begin <= l1->x_begin && l2->x_end >= l1->x_end) {
                    result.data->push_back(*l1);
                } else if (l2->x_begin <= l1->x_begin && l2->x_end <= l1->x_end) {
                    Repeat tmp(l1->x_begin, l1->y_begin, l2->x_end, l2->y_end, l2->x_end - l1->x_begin + 1);
                    result.data->push_back(tmp);
                }
            }
    result.x_limit_left   = listRepeats1.x_limit_left;
    result.x_limit_right  = listRepeats1.x_limit_right;
    result.y_limit_above  = listRepeats1.y_limit_above;
    result.y_limit_bottom = listRepeats1.y_limit_bottom;
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
