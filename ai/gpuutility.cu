#ifdef CUDA_EDIT
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<<__VA_ARGS__>>>
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Kernel function to add the elements of two arrays
__global__ void printKernel()
{
	printf("Hello from mykernel\n");
}

void testPrint()
{
	printKernel CUDA_KERNEL(1,1) ();
	cudaDeviceSynchronize();
}
