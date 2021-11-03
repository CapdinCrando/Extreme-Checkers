#include <iostream>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
	y[i] = x[i] + y[i];
}
