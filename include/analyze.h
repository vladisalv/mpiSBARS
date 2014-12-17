#ifndef __ANALYZE_HEADER__
#define __ANALYZE_HEADER__

#include "myMPI.h"
#include "gpu_computing.h"
#include "matrix_gomology.h"
#include "matrix_analysis.h"

#include <list>

using namespace std;

struct Repeat {
    long x_begin, y_begin;
    long x_end, y_end;
    long length;
    Repeat();
    Repeat(long x1, long y1, long x2, long y2, long len);
    void Print();
};

typedef Repeat Repeat_type;

class Analyze {
    MyMPI me;

    MatrixGomology matrix;
    ulong length;
    ulong my_global_height;
    bool use_gpu;
    MatrixAnalysis result_matrix;

    list<struct Repeat> repeat_answer;
    list<struct Repeat> repeat_unknow;
    list<struct Repeat> repeat_begin;

    struct Repeat *repeat_alien;
    ulong repeat_alien_size;

    bool *flag_end, *flag_end_prev;
    long *flag_begin;

    MPI_Aint lb, extent;
    MPI_Win win_flag_end, win_flag_begin, win_repeat_alien;
    MPI_Group group_comm_world, group_prev, group_next, group_prev_all, group_next_all;


    void localSearch0();
    void localSearch1_n();
    Repeat findRepeat(ulong y, ulong x);

    void analysisRepeatBegin();
    void analysisRepeatAlien();
    void analysisRepeatUnknow();
    Repeat requestRepeat(int rank, ulong x);
    void formResultMatrix();
    bool searchRepeat(ulong y, ulong x);
    void initFlagEnd();
    void initFlagEndPrev();
public:
    Analyze(MyMPI me);
    ~Analyze();

    MatrixAnalysis doAnalyze(MatrixGomology matrixGomology, ulong length, GpuComputing gpu);
};

#endif /* __ANALYZE_HEADER__ */
