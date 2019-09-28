//============================================================================
// Name        : CudaMap.cu
// Author      : Hang
//============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>

#include <ctime>
#include <cstdio>

#include <iostream>
using namespace std;

__global__ void addTen(float* d, int count) {

  int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;

  // Thread position in the block
  int threadPosInBlock = threadIdx.x + blockDim.x * threadIdx.y +
		                 blockDim.x * blockDim.y * threadIdx.z;

  // Block position in grid
  int blockPosInGrid = blockIdx.x + gridDim.x * blockIdx.y +
		               gridDim.x * gridDim.y * blockIdx.z;

  // Final thread ID
  int tid = blockPosInGrid * threadsPerBlock + threadPosInBlock;

  if (tid < count) {
	  d[tid] = d[tid] + 10;
  }

}

int main(void) {

  curandGenerator_t gen;

  // Initialize generator
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);

  // Seed value with current time
  curandSetPseudoRandomGeneratorSeed(gen, time(0));

  cudaError_t status;

  const int count = 123456;
  const int size = count * sizeof(float);

  float* d;
  float h[count];

  cudaMalloc(&d, size);
  curandGenerateUniform(gen, d, count);

  dim3 block(8,8,8); // 8x8x8=512, 123456/512=241
  dim3 grid(16,16);  // 16x16=256

  addTen<<<grid,block>>>(d, count);

  status = cudaMemcpy(h,d,size,cudaMemcpyDeviceToHost);

  cudaFree(d);

  for (int i = 0; i < 10; ++i) {
	  cout << h[i] << '\n';
  }

  return 0;

}






