#include "compare_gpu.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

__global__ void kernel(TypeDecomposition *dec1, ulong length_decompose1,
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

void compareDecompositionGpu(TypeDecomposition *decomposeHost1, ulong length_decompose1,
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

    kernel<<< 128, 128 >>>
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
