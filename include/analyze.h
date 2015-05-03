#ifndef __ANALYZE_HEADER__
#define __ANALYZE_HEADER__

#include "myMPI.h"
#include "gpu_computing.h"

#include "decomposition.h"
#include "compare.h"
#include "matrix_gomology.h"
#include "list_repeats.h"

#include <math.h>

class Analyze {
    MyMPI me;
    GpuComputing gpu;

    double eps;
    ulong min_length;
    double fidelity_repeat;
    size_t limit_memory;

    TypeDecomposition *buf_tmp;
    Decomposition decomposition, dec_other;
    MPI_Request req;
    void recvDecompositon();
    bool recvDecompositonAsync();
    void waitDecomposition();
    int source_proc;

public:
    Analyze(MyMPI me, GpuComputing gpu, double eps, ulong min_length, double fidelity_repeat, size_t limit_memory);
    ~Analyze();

    ListRepeats doAnalyze(MatrixGomology matrixGomology);
    ListRepeats doAnalyze(Decomposition decomposition);
    ListRepeats doAnalyze(Decomposition decomposition1, Decomposition decomposition2);
    ListRepeats comparisonRepeats(ListRepeats listRepeats1, ListRepeats listRepeats2);

    double getEps();
    ulong  getMinLengthRepeat();
    double getFidelityRepeat();
    size_t getLimitMemoryMatrix();

    void setEps(double eps_new);
    void setMinLengthRepeat(ulong min_length_new);
    void setFidelityRepeat(double fidelity_repeat_new);
    void setLimitMemoryMatrix(size_t limit_memory_new);
};

#endif /* __ANALYZE_HEADER__ */
