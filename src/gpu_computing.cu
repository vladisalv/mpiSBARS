#include "gpu_computing.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

GpuComputing::GpuComputing()
{
}

GpuComputing::GpuComputing(MyMPI new_me, bool use)
    : me(new_me), use_gpu(use)
{
}

GpuComputing::~GpuComputing()
{
}

bool GpuComputing::isUse()
{
    return use_gpu;
}

__global__ void kernelDecompose(TypeDecomposition *decompose, uint number_coef, TypeProfile *profile,
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


__global__ void kernelDivide(TypeDecomposition *decompose, ulong window_size, ulong length_decompose)
{
    double k = window_size / 2;
    ulong i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < length_decompose) {
        decompose[i] /= k;
        i += gridDim.x * blockDim.x;
    }
}


void GpuComputing::doDecomposeGPU(TypeDecomposition *decomposeHost, uint number_window, uint number_coef,
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

    kernelDecompose<<< 128, 128 >>>(decomposeDevice, number_coef, profileDevice, step, length_profile, window_size);
    kernelDivide<<< 128, 128 >>>(decomposeDevice, window_size, length_decompose);

    HANDLE_ERROR(cudaMemcpy(decomposeHost, decomposeDevice, size_decompose, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(decomposeDevice));
    HANDLE_ERROR(cudaFree(profileDevice));
}

__global__ void kernelCompare(TypeDecomposition *dec1, ulong length_decompose1,
                       TypeDecomposition *dec2, ulong length_decompose2,
                       TypeGomology *result, ulong width, double eps)
{
    ulong offset_dec1 = blockIdx.x;
    ulong offset_dec2;
    TypeDecomposition sum, difference;
    while (offset_dec1 < length_decompose1) {
        offset_dec2 = threadIdx.x;
        while (offset_dec2 < length_decompose2) {
            sum = 0.;
            for (int i = 0; i < width; i++) {
                difference = dec1[offset_dec1 * width + i] - dec2[length_decompose2 * i + offset_dec2];
                sum += difference * difference;
            }
            if (sum > eps)
                result[offset_dec1 * length_decompose2 + offset_dec2] = false;
            else
                result[offset_dec1 * length_decompose2 + offset_dec2] = true;
            offset_dec2 += blockDim.x;
        }
        offset_dec1 += gridDim.x;
    }
}

void GpuComputing::compareDecompositionGpu(TypeDecomposition *decomposeHost1, ulong length_decompose1,
                             TypeDecomposition *decomposeHost2, ulong length_decompose2,
                             ulong width, TypeGomology *resultHost, ulong begin,
                             ulong sum_all, double eps)
{
    size_t size_decompose1 = length_decompose1 * width * sizeof(TypeDecomposition);
    size_t size_decompose2 = length_decompose2 * width * sizeof(TypeDecomposition);
    size_t size_result = length_decompose1 * length_decompose2 * sizeof(TypeGomology);

    TypeDecomposition *decomposeDevice1, *decomposeDevice2;
    TypeGomology *resultDevice;
    HANDLE_ERROR(cudaMalloc((void **)&decomposeDevice1, size_decompose1));
    HANDLE_ERROR(cudaMalloc((void **)&decomposeDevice2, size_decompose2));
    HANDLE_ERROR(cudaMalloc((void **)&resultDevice, size_result));

    TypeDecomposition *decomposeHost2Transpose = new TypeDecomposition [length_decompose2 * width];
    for (int i = 0; i < width; i++)
        for (int j = 0; j < length_decompose2; j++)
            decomposeHost2Transpose[i * length_decompose2 + j] = decomposeHost2[j * width + i];

    HANDLE_ERROR(cudaMemcpy(decomposeDevice1, decomposeHost1,          size_decompose1, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(decomposeDevice2, decomposeHost2Transpose, size_decompose2, cudaMemcpyHostToDevice));

    kernelCompare<<< 128, 128 >>>
        (decomposeDevice1, length_decompose1,
         decomposeDevice2, length_decompose2,
         resultDevice, width, eps);

    for (int i = 0; i < length_decompose1; i++)
        HANDLE_ERROR(cudaMemcpy(&resultHost[i * sum_all + begin],
                     &resultDevice[i * length_decompose2],
                     length_decompose2 * sizeof(TypeGomology), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(decomposeDevice1));
    HANDLE_ERROR(cudaFree(decomposeDevice2));
    HANDLE_ERROR(cudaFree(resultDevice));
    free(decomposeHost2Transpose);
}
