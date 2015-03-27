#ifndef __ANALYZE_HEADER__
#define __ANALYZE_HEADER__

#include "myMPI.h"
#include "gpu_computing.h"

#include "decomposition.h"
#include "matrix_gomology.h"
#include "set_repeats.h"

#include <list>

using namespace std;

class Analyze {
    MyMPI me;
    GpuComputing gpu;

    double eps;
    ulong min_length;
    double fidelity_repeat;
    size_t limit_memory;

    MatrixGomology matrix;
    ulong my_global_height;

    list<struct Repeat> repeat_answer;
    list<struct Repeat> repeat_unknow;
    list<struct Repeat> repeat_begin;

    bool *flag_end, *flag_end_prev;

    struct Repeat *repeat_alien;
    ulong repeat_alien_size;

    MPI_Aint lb, extent;
    MPI_Win win_flag_end, win_repeat_alien;
    MPI_Group group_comm_world, group_prev, group_next, group_prev_all, group_next_all;


    Repeat findRepeat(Coordinates cor);
    void localSearch0();
    void localSearch1_n();
    Coordinates searchRepeat(Coordinates cor);

    void analysisRepeatBegin();
    void formRepeatAlien();
    void analysisRepeatUnknow();
    Repeat requestRepeat(int rank, ulong x);
    void sortRepeatAnswer();
public:
    Analyze(MyMPI me, GpuComputing gpu, double eps, ulong min_length, double fidelity_repeat, size_t limit_memory);
    ~Analyze();

    SetRepeats doAnalyze(MatrixGomology matrixGomology);
    SetRepeats doAnalyze(Decomposition decomposition);
    SetRepeats doAnalyze(Decomposition decomposition1, Decomposition decomposition2);
    SetRepeats comparisonRepeats(SetRepeats setRepeats1, SetRepeats setRepeats2);

    double getEps();
    ulong getMinLengthRepeat();
    double getFidelityRepeat();
    size_t getLimitMemoryMatrix();

    void setEps(double eps_new);
    void setMinLengthRepeat(ulong min_length_new);
    void setFidelityRepeat(double fidelity_repeat_new);
    void setLimitMemoryMatrix(size_t limit_memory_new);
};

#endif /* __ANALYZE_HEADER__ */
