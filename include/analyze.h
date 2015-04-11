#ifndef __ANALYZE_HEADER__
#define __ANALYZE_HEADER__

#include "myMPI.h"
#include "gpu_computing.h"

#include "decomposition.h"
#include "matrix_gomology.h"
#include "set_repeats.h"

#include <math.h>

class Analyze {
    MyMPI me;
    GpuComputing gpu;

    double eps;
    ulong min_length;
    double fidelity_repeat;
    size_t limit_memory;

    TypeDecomposition *dec_other, *buf_tmp;
    MPI_Request req;
    void recvDecompositon();
    bool recvDecompositonAsync();
    void waitDecomposition();

public:
    Analyze(MyMPI me, GpuComputing gpu, double eps, ulong min_length, double fidelity_repeat, size_t limit_memory);
    ~Analyze();

    SetRepeats doAnalyze(MatrixGomology matrixGomology);
    SetRepeats doAnalyze(Decomposition decomposition);
    SetRepeats doAnalyze(Decomposition decomposition1, Decomposition decomposition2);
    SetRepeats comparisonRepeats(SetRepeats setRepeats1, SetRepeats setRepeats2);

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
