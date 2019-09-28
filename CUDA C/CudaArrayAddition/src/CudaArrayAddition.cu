//============================================================================
// Name        : CudaArrayAddition.cpp
// Author      : Hang
//============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

void check(cudaError_t e) {
	if (e != cudaSuccess) {
		printf(cudaGetErrorString(e));
	}
}

// Define kernel, run on GPU, called from CPU
__global__ void addArraysGPU(int* a, int* b, int* c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

int main(void)
{
  const int count = 5;

  // Define host arrays to add
  int ha[] = { 1, 2, 3, 4, 5 };
  int hb[] = { 10, 20, 30, 40, 50 };
  int hc[count];

  // Allocate GPU device memory
  int *da, *db, *dc;
  int size = sizeof(int)*count;

  cudaMalloc(&da, size);
  cudaMalloc(&db, size);
  cudaMalloc(&dc, size);

  // Copy data from host memory to device memory
  cudaMemcpy(da,ha,size,cudaMemcpyHostToDevice);
  cudaMemcpy(db,hb,size,cudaMemcpyHostToDevice);

  // Call function to run on GPU, 1: one block, count: 1 block has "count" threads
  addArraysGPU<<<1,count>>>(da,db,dc);

  // Copy result from device memory to host memory
  cudaMemcpy(hc,dc,size,cudaMemcpyDeviceToHost);

  // Print result
  printf("%d %d %d %d %d",
	  hc[0],
	  hc[1],
	  hc[2],
	  hc[3],
	  hc[4]);

  // Release GPU memory
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  return 0;

}
