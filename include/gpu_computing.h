#ifndef __GPU_COMPUTING_HEADER__
#define __GPU_COMPUTING_HEADER__

#include "myMPI.h"

class GpuComputing {
    MyMPI me;
    bool use_gpu;
public:
    GpuComputing();
    GpuComputing(MyMPI me, bool use_gpu);
    ~GpuComputing();

    void doDecomposeGPU(TypeDecomposition *dataHost, uint number_window, uint number_coef,
                    TypeProfile *profileHost, uint window, uint step);
    void compareDecompositionGpu(TypeDecomposition *decompose1, ulong length_decompose1,
                              TypeDecomposition *decompose2, ulong length_decompose2,
                              ulong width, TypeGomology *data, ulong begin,
                              ulong sum_all, double eps);

    bool isUse();

    void debugInfo();
};

#endif /* __GPU_COMPUTING_HEADER__ */
