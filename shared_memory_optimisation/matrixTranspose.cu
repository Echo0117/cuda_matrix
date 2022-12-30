
/**
  * Kernel 1: Implement a first kernel where each block (using BSXY x BSXY threads) transposes a BSXY x BSXY tile of A, and writes it into the corresponding location in At. Do without using shared memory.
  *
  * Kernel 2: In the second kernel, do the same, but using the shared memory. Each block should load a tile of BSXY x BSXY of A into the shared memory, then perform the transposition using this tile in the shared memory into At. Test the difference in speedup. Test the performance using shared memory without padding and with padding (to avoid shared memory bank conflicts).
  *
  * Kernel 3: In this kernel, perform the transpose in-place on the matrix A (do not use At). A block should be transpose two tiles simultenously to be able to do this.
  *
  */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "cuda.h"
#include <cfloat>

#define BSXY 32
// #define N 1024

float *A, *At, *Agpu;
float *dA, *dAt;

__global__ void transposesMatrixGPUB(float *dA, float *dAt, int N)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && j < N) {
      dAt[j * N + i] = dA[i * N + j];
  }
}

__global__ void transposesMatrixGPUBSharedMemory(float *dA, float *dAt, int N)
{
  __shared__ float shA[BSXY][BSXY];
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  shA[threadIdx.x][threadIdx.y] = dA[i + j * N];
  __syncthreads();
  if (i < N && j < N) {
      dAt[j * N + i] = shA[threadIdx.x][threadIdx.y];
  }
  __syncthreads();
}

__global__ void transposesMatrixGPUBInPlace(float *dA, int N)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && j < N) {
     float temp = dA[j * N + i];
     dA[j * N + i] = dA[i * N + j];
     dA[i * N + j] = temp;
  }
}

void transposeCPU(float *A, float *At, int N)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      At[i * N + j] = A[j * N + i];
    }
  }
}


void verifyResults(float *At, int N)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (At[i + j * N] != Agpu[i + j * N]) {
        std::cout << "incorrect A[" << i << "][" << j << "]" << std::endl;
        return;
      }
    }
  }
  std::cout << "correct!" << std::endl;
}

int main()
{
  // Allocate A and At
  // A is an N * N matrix stored by rows, i.e. A(i, j) = A[i * N + j]
  // At is also stored by rows and is the transpose of A, i.e., At(i, j) = A(j, i)
  int N = 128;
  A = (float *) malloc(N * N * sizeof(A[0]));
  At = (float *) malloc(N * N * sizeof(At[0]));
  Agpu = (float *) malloc (N * N * sizeof(Agpu[0]));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i + j * N] = (float)i * (float)j;
      At[i * N + j] = A[j * N + i];
      Agpu[i + j * N] = 0.0f;
    }
  }
  
  // Allocate dA and dAt, and call the corresponding matrix transpose kernel
  transposeCPU(A, Agpu, N);
  cudaMalloc(&dA, sizeof(dA[0]) * N * N);
  cudaMalloc(&dAt, sizeof(dAt[0]) * N * N);
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dAt, At, N * N * sizeof(float), cudaMemcpyHostToDevice);

  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x = N / BSXY;
    dimGrid.y = N;
    dimGrid.z = 1;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    transposesMatrixGPUB<<<dimGrid, dimBlock>>>(dA, dAt, N);
  }

  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x = N / BSXY;
    dimGrid.y = N;
    dimGrid.z = 1;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    // transposesMatrixGPUBSharedMemory<<<dimGrid, dimBlock>>>(dA, dAt, N);
  }

  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x = N / BSXY;
    dimGrid.y = N;
    dimGrid.z = 1;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    // transposesMatrixGPUBInPlace<<<dimGrid, dimBlock>>>(dA, N);
  }

  cudaMemcpy(Agpu, dAt, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  verifyResults(At, N);

  // Deallocate dA and dAt
  cudaFree(dA); cudaFree(dAt); 

  // Deallocate A and At
  free(A);
  free(At);
  return 0;
}
