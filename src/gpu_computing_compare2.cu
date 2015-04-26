#ifdef USE_CUDA

#include "gpu_computing.h"


#define get_elem(array, Row, Column) \
(((TypeDecomposition*)((char*)array.ptr + (Row) * array.pitch))[(Column)])

#define get_addr(array, Row, Column) \
((TypeGomology&)(((TypeGomology*)((char*)array.ptr + (Row) * array.pitch))[(Column)]))


__global__ void kernelSharedFast4(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C, double eps)
{
	int y = threadIdx.y + blockIdx.y * 64;
	int x = threadIdx.x + blockIdx.x * 128;

	__shared__ TypeDecomposition a[64][32];
	__shared__ TypeDecomposition b[32][128];

    float result[16];
    result[0] = result[1] = result[2] = result[3] = result[4] = result[5] =
    result[6] = result[7] = result[8] = result[9] = result[10] = result[11] =
    result[12] = result[13] = result[14] = result[15] = 0.;

	int num_iter = A.xsize / blockDim.x;
	for (int i = 0; i < num_iter; i++) {
		a[threadIdx.y +  0][threadIdx.x] = get_elem(A, y +  0, blockDim.x * i + threadIdx.x);
		a[threadIdx.y + 16][threadIdx.x] = get_elem(A, y + 16, blockDim.x * i + threadIdx.x);
		a[threadIdx.y + 32][threadIdx.x] = get_elem(A, y + 32, blockDim.x * i + threadIdx.x);
		a[threadIdx.y + 48][threadIdx.x] = get_elem(A, y + 48, blockDim.x * i + threadIdx.x);

		b[threadIdx.y +  0][threadIdx.x +  0] = get_elem(B, blockDim.x * i + threadIdx.y +  0, x +  0);
		b[threadIdx.y + 16][threadIdx.x +  0] = get_elem(B, blockDim.x * i + threadIdx.y + 16, x +  0);
		b[threadIdx.y +  0][threadIdx.x + 32] = get_elem(B, blockDim.x * i + threadIdx.y +  0, x + 32);
		b[threadIdx.y + 16][threadIdx.x + 32] = get_elem(B, blockDim.x * i + threadIdx.y + 16, x + 32);
		b[threadIdx.y +  0][threadIdx.x + 64] = get_elem(B, blockDim.x * i + threadIdx.y +  0, x + 64);
		b[threadIdx.y + 16][threadIdx.x + 64] = get_elem(B, blockDim.x * i + threadIdx.y + 16, x + 64);
		b[threadIdx.y +  0][threadIdx.x + 96] = get_elem(B, blockDim.x * i + threadIdx.y +  0, x + 96);
		b[threadIdx.y + 16][threadIdx.x + 96] = get_elem(B, blockDim.x * i + threadIdx.y + 16, x + 96);

		__syncthreads();
		TypeDecomposition b_tmp, diff, a0, a16, a32, a48;
		for (int k = 0; k < blockDim.x; k++) {
            a0  = a[threadIdx.y +  0][k];
            a16 = a[threadIdx.y + 16][k];
            a32 = a[threadIdx.y + 32][k];
            a48 = a[threadIdx.y + 48][k];

			b_tmp = b[k][threadIdx.x  +  0];
			diff  = a0  - b_tmp;
            result[0]  += diff * diff;
			diff  = a16 - b_tmp;
            result[1]  += diff * diff;
			diff  = a32 - b_tmp;
            result[2]  += diff * diff;
			diff  = a48 - b_tmp;
            result[3]  += diff * diff;

			b_tmp = b[k][threadIdx.x  + 32];
			diff  = a0  - b_tmp;
            result[4]  += diff * diff;
			diff  = a16 - b_tmp;
            result[5]  += diff * diff;
			diff  = a32 - b_tmp;
            result[6]  += diff * diff;
			diff  = a48 - b_tmp;
            result[7]  += diff * diff;

			b_tmp = b[k][threadIdx.x  + 64];
			diff  = a0  - b_tmp;
            result[8]   += diff * diff;
			diff  = a16 - b_tmp;
            result[9]   += diff * diff;
			diff  = a32 - b_tmp;
            result[10]  += diff * diff;
			diff  = a48 - b_tmp;
            result[11]  += diff * diff;

			b_tmp = b[k][threadIdx.x  + 96];
			diff  = a0  - b_tmp;
            result[12]  += diff * diff;
			diff  = a16 - b_tmp;
            result[13]  += diff * diff;
			diff  = a32 - b_tmp;
            result[14]  += diff * diff;
			diff  = a48 - b_tmp;
            result[15]  += diff * diff;
        }
		__syncthreads();
	}
    get_addr(C, y +  0, x +  0) = result[0]  > eps ? false : true;
    get_addr(C, y + 16, x +  0) = result[1]  > eps ? false : true;
    get_addr(C, y + 32, x +  0) = result[2]  > eps ? false : true;
    get_addr(C, y + 48, x +  0) = result[3]  > eps ? false : true;
    get_addr(C, y +  0, x + 32) = result[4]  > eps ? false : true;
    get_addr(C, y + 16, x + 32) = result[5]  > eps ? false : true;
    get_addr(C, y + 32, x + 32) = result[6]  > eps ? false : true;
    get_addr(C, y + 48, x + 32) = result[7]  > eps ? false : true;
    get_addr(C, y +  0, x + 64) = result[8]  > eps ? false : true;
    get_addr(C, y + 16, x + 64) = result[9]  > eps ? false : true;
    get_addr(C, y + 32, x + 64) = result[10] > eps ? false : true;
    get_addr(C, y + 48, x + 64) = result[11] > eps ? false : true;
    get_addr(C, y +  0, x + 96) = result[12] > eps ? false : true;
    get_addr(C, y + 16, x + 96) = result[13] > eps ? false : true;
    get_addr(C, y + 32, x + 96) = result[14] > eps ? false : true;
    get_addr(C, y + 48, x + 96) = result[15] > eps ? false : true;
}

void GpuComputing::compareDecompositionGpu2(TypeDecomposition *decomposeHost1, ulong length_decompose1,
                             TypeDecomposition *decomposeHost2, ulong length_decompose2,
                             ulong width, TypeGomology *resultHost, ulong begin,
                             ulong sum_all, double eps)
{
    TypeDecomposition *decomposeHost2Transpose = new TypeDecomposition [length_decompose2 * width];
    for (int i = 0; i < width; i++)
        for (int j = 0; j < length_decompose2; j++)
            decomposeHost2Transpose[i * length_decompose2 + j] = decomposeHost2[j * width + i];
    decomposeHost2 = decomposeHost2Transpose;

	int blockSizeX = 128, blockSizeY = 64;
    cudaPitchedPtr hostA, hostB, hostC;
    hostA = make_cudaPitchedPtr(decomposeHost1, width * sizeof(TypeDecomposition), width, length_decompose1);
    hostB = make_cudaPitchedPtr(decomposeHost2, length_decompose2 * sizeof(TypeDecomposition), length_decompose2, width);
    hostC = make_cudaPitchedPtr(&resultHost[begin], sum_all * sizeof(TypeGomology), hostB.xsize, hostA.ysize);

	int newWidthA, newHeightA, newWidthB, newHeightB, newWidthC, newHeightC;
	newWidthA  = blockSizeX * ((hostA.xsize + blockSizeX - 1) / blockSizeX);
	newHeightA = blockSizeY * ((hostA.ysize + blockSizeY - 1) / blockSizeY);
	newWidthB  = blockSizeX * ((hostB.xsize + blockSizeX - 1) / blockSizeX);
	newHeightB = blockSizeY * ((hostB.ysize + blockSizeY - 1) / blockSizeY);
	newWidthC  = blockSizeX * ((hostC.xsize + blockSizeX - 1) / blockSizeX);
	newHeightC = blockSizeY * ((hostC.ysize + blockSizeY - 1) / blockSizeY);

    cudaPitchedPtr devA, devB, devC;
    devA = make_cudaPitchedPtr(0, 0, newWidthA, newHeightA);
    devB = make_cudaPitchedPtr(0, 0, newWidthB, newHeightB);
    devC = make_cudaPitchedPtr(0, 0, newWidthC, newHeightC);

    HANDLE_ERROR(cudaMallocPitch((void **)&devA.ptr, &devA.pitch, devA.xsize * sizeof(TypeDecomposition), devA.ysize));
    HANDLE_ERROR(cudaMallocPitch((void **)&devB.ptr, &devB.pitch, devB.xsize * sizeof(TypeDecomposition), devB.ysize));
    HANDLE_ERROR(cudaMallocPitch((void **)&devC.ptr, &devC.pitch, devC.xsize * sizeof(TypeGomology), devC.ysize));

	HANDLE_ERROR(cudaMemset(devA.ptr, 0, devA.pitch * devA.ysize));
	HANDLE_ERROR(cudaMemset(devB.ptr, 0, devB.pitch * devB.ysize));
	HANDLE_ERROR(cudaMemset(devC.ptr, 0, devC.pitch * devC.ysize));

    HANDLE_ERROR(cudaMemcpy2D(devA.ptr, devA.pitch,
							hostA.ptr, hostA.pitch,
							hostA.pitch, hostA.ysize,
							cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(devB.ptr, devB.pitch,
							hostB.ptr, hostB.pitch,
							hostB.pitch, hostB.ysize,
							cudaMemcpyHostToDevice));

    dim3 block(32, 16);
    dim3 grid(devC.xsize / blockSizeX, devC.ysize / blockSizeY);

    me.rootMessage("GPU compare version 2\n");
    kernelSharedFast4<<< grid, block >>>(devA, devB, devC, eps);

    HANDLE_ERROR(cudaMemcpy2D(hostC.ptr, hostC.pitch,
							devC.ptr, devC.pitch,
							hostC.xsize * sizeof(TypeGomology), hostC.ysize,
							cudaMemcpyDeviceToHost));

    /*
    for (int i = 0; i < length_decompose1; i++)
        HANDLE_ERROR(cudaMemcpy(&resultHost[i * sum_all + begin],
                     &((TypeGomology)devC.ptr)[i * length_decompose2],
                     length_decompose2 * sizeof(TypeGomology), cudaMemcpyDeviceToHost));
        */

    HANDLE_ERROR(cudaFree(devA.ptr));
    HANDLE_ERROR(cudaFree(devB.ptr));
    HANDLE_ERROR(cudaFree(devC.ptr));
    free(decomposeHost2Transpose);
}

#endif /* USE_CUDA */
