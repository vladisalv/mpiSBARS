#ifdef USE_CUDA

#include "gpu_computing.h"

void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(0);
    }
}

GpuComputing::GpuComputing(MyMPI new_me, bool use)
    : me(new_me), use_gpu(use), decomposition1(0), decomposition2(0)
{
    int deviceCount;
    HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
    int myDevice;
    if (me.getSize() < 4)
        myDevice = me.getRank() % deviceCount + 1;
    else
        myDevice = me.getRank() % deviceCount;
	myDevice = 3;
    HANDLE_ERROR(cudaSetDevice(myDevice));
}

GpuComputing::~GpuComputing()
{
}

bool GpuComputing::isUse()
{
    return use_gpu;
}

void GpuComputing::infoDevices()
{
    int count;
    HANDLE_ERROR( cudaGetDeviceCount( &count ) );
    for (int i=0; i< count; i++) {
		printInfoDevice(i);
    }
}

void GpuComputing::infoMyDevice()
{
	printInfoDevice(myId);
}

void GpuComputing::setDevice(int major, int minor)
{
	cudaDeviceProp prop;
	memset(&prop, 0, sizeof(cudaDeviceProp));

	prop.major = major;
	prop.minor = minor;

	HANDLE_ERROR(cudaChooseDevice(&myId, &prop));
	HANDLE_ERROR(cudaSetDevice(myId));
}

void GpuComputing::printInfoDevice(int i)
{
    cudaDeviceProp  prop;
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
    printf( "   --- General Information for device %d ---\n", i );
    printf( "Name:  %s\n", prop.name );
    printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
    printf( "Clock rate:  %d\n", prop.clockRate );
    printf( "Device copy overlap:  " );
    if (prop.deviceOverlap)
        printf( "Enabled\n" );
    else
        printf( "Disabled\n");
    printf( "Kernel execution timeout :  " );
    if (prop.kernelExecTimeoutEnabled)
        printf( "Enabled\n" );
    else
        printf( "Disabled\n" );

    printf( "   --- Memory Information for device %d ---\n", i );
    printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
    printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
    printf( "Max mem pitch:  %ld\n", prop.memPitch );
    printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

    printf( "   --- MP Information for device %d ---\n", i );
    printf( "Multiprocessor count:  %d\n",
                prop.multiProcessorCount );
    printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
    printf( "Registers per mp:  %d\n", prop.regsPerBlock );
    printf( "Threads in warp:  %d\n", prop.warpSize );
    printf( "Max threads per block:  %d\n",
                prop.maxThreadsPerBlock );
    printf( "Max thread dimensions:  (%d, %d, %d)\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                prop.maxThreadsDim[2] );
    printf( "Max grid dimensions:  (%d, %d, %d)\n",
                prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2] );
    printf( "\n" );
}

void GpuComputing::debugInfo(const char *file, int line, const char *info)
{
    printf("This is debugInfo(%s) of %s in %s at line %d\n", info, "GpuComputing", file, line);
}

#endif /* USE_CUDA */
