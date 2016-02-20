#ifndef __GPU_COMPUTING_HEADER__
#define __GPU_COMPUTING_HEADER__

#include "types.h"
#include "myMPI.h"

#include <stdlib.h> // exit()

#ifdef USE_CUDA

#include <cuda_runtime.h>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
void HandleError(cudaError_t err, const char *file, int line);

#endif /* USE_CUDA */


class GpuComputing {
    MyMPI me;
    bool use_gpu;
    int myId;
	void printInfoDevice(int id);

    TypeDecomposition *decomposition1, *decomposition2;
public:
    GpuComputing(MyMPI me, bool use_gpu);
    ~GpuComputing();

    void doDecomposeGPU(TypeDecomposition *dataHost, uint number_window, uint number_coef,
                    TypeProfile *profileHost, uint window, uint step);
    void compareDecompositionGpu(TypeDecomposition *decompose1, ulong length_decompose1,
                              TypeDecomposition *decompose2, ulong length_decompose2,
                              ulong width, TypeGomology *data, ulong begin,
                              ulong sum_all, double eps);
    void compareDecompositionGpu2(TypeDecomposition *decompose1, ulong length_decompose1,
                              TypeDecomposition *decompose2, ulong length_decompose2,
                              ulong width, TypeGomology *data, ulong begin,
                              ulong sum_all, double eps);

    bool isUse();

	void infoDevices();
	void infoMyDevice();
	void setDevice(int major, int minor);

    void debugInfo(const char *file, int line, const char *info = 0);
};

#endif /* __GPU_COMPUTING_HEADER__ */
