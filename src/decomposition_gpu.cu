#include "decomposition_gpu.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

__global__ void kernel(TypeDecomposition *decompose, uint number_coef, TypeProfile *profile,
                       uint step, uint length_profile, uint window_size)
{
    TypeDecomposition p1, p2, p3;
    TypeDecomposition q1, q2, q3;
    TypeDecomposition yi, ti, costi;

    // init offset
    uint offset_window;
    ulong offset_profile;
    ulong offset_block = blockIdx.x * step;
    ulong offset_decompose = number_coef * blockIdx.x;

    while (offset_block < length_profile - blockDim.x) {
        offset_window = threadIdx.x;
        while (offset_window < window_size) {
            offset_profile = offset_block + offset_window;
            yi = (TypeDecomposition)profile[offset_profile];
            ti = M_PI * (2 * offset_window + 1 - window_size) / window_size;
            costi = 2 * cos(ti);

            p1 = 0;
            p2 = yi * sin(ti);
            q1 = yi;
            q2 = yi * costi / 2.0;

            atomicAdd(&(decompose[offset_decompose]), yi / SQRT_2);

            for (uint j = 1; j < number_coef; j++) {
                if (j & 1) {
                    p3 = p2;
                    p2 = p1;
                    p1 = costi * p2 - p3;
                    atomicAdd(&(decompose[offset_decompose + j]), p1);
                } else {
                    q3 = q2;
                    q2 = q1;
                    q1 = costi * q2 - q3;
                    atomicAdd(&(decompose[offset_decompose + j]), q1);
                }
            }

            offset_window += blockDim.x;
        }
        offset_block += gridDim.x * step;
        offset_decompose += gridDim.x * number_coef;
    }
}


__global__ void kernel_divide(TypeDecomposition *decompose, ulong window_size, ulong length_decompose)
{
    double k = window_size / 2;
    ulong i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < length_decompose) {
        decompose[i] /= k;
        i += gridDim.x * blockDim.x;
    }
}

void doDecomposeGPU(TypeDecomposition *decomposeHost, uint number_window, uint number_coef,
                    TypeProfile *profileHost, uint window_size, uint step)
{
    ulong length_profile = step * (number_window - 1) + window_size;
    ulong length_decompose = number_window * number_coef;
    size_t size_profile = length_profile * sizeof(TypeProfile);
    size_t size_decompose = length_decompose * sizeof(TypeDecomposition);

    TypeDecomposition *decomposeDevice;
    TypeProfile *profileDevice;
    HANDLE_ERROR(cudaMalloc((void **)&decomposeDevice, size_decompose));
    HANDLE_ERROR(cudaMalloc((void **)&profileDevice, size_profile));

    HANDLE_ERROR(cudaMemset(decomposeDevice, 0, size_decompose));
    HANDLE_ERROR(cudaMemcpy(profileDevice, profileHost, size_profile, cudaMemcpyHostToDevice));

    kernel<<< 128, 128 >>>(decomposeDevice, number_coef, profileDevice, step, length_profile, window_size);
    kernel_divide<<< 128, 128 >>>(decomposeDevice, window_size, length_decompose);

    HANDLE_ERROR(cudaMemcpy(decomposeHost, decomposeDevice, size_decompose, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(decomposeDevice));
    HANDLE_ERROR(cudaFree(profileDevice));
}
