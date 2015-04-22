#ifndef USE_CUDA

#include "gpu_computing.h"


GpuComputing::GpuComputing()
{
}

GpuComputing::GpuComputing(MyMPI new_me, bool use)
    : me(new_me), use_gpu(false)
{
}

GpuComputing::~GpuComputing()
{
}

bool GpuComputing::isUse()
{
    return use_gpu;
}


void GpuComputing::doDecomposeGPU(TypeDecomposition *decomposeHost, uint number_window, uint number_coef,
                    TypeProfile *profileHost, uint window_size, uint step)
{
}


void GpuComputing::compareDecompositionGpu(TypeDecomposition *decomposeHost1, ulong length_decompose1,
                             TypeDecomposition *decomposeHost2, ulong length_decompose2,
                             ulong width, TypeGomology *resultHost, ulong begin,
                             ulong sum_all, double eps)
{
}

void GpuComputing::printInfoDevice(int id)
{
    printf("You do not use GPU device\n");
}

void GpuComputing::infoDevices()
{
}

void GpuComputing::infoMyDevice()
{
}

void GpuComputing::setDevice(int major, int minor)
{
}

void GpuComputing::debugInfo(const char *file, int line, const char *info)
{
    printf("This is debugInfo(%s) of %s in %s at line %d\n", info, "GpuComputing", file, line);
}
#endif /* USE_CUDA */
