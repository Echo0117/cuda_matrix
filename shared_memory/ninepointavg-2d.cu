/**
  * In this exercise, we will implement GPU kernels for computing the average of 9 points on a 2D array.
  * Dans cet exercice, nous implantons un kernel GPU pour un calcul de moyenne de 9 points sur un tableau 2D.
  *
  * Kernel 1: Use 1D blocks, no additional threads (1 thread per block)
  * Kernel 1: Utiliser blocs de 1D, pas de threads (1 thread par bloc)
  *
  * Kernel 2: Use 2D blocks, no additional threads (1 thread per block)
  * Kernel 2: Utiliser blocs de 2D, pas de threads (1 thread par bloc)
  *
  * Kernel 3: Use 2D blocks and 2D threads, each thread computing 1 element of Aavg
  * Kernel 3: Utiliser blocs de 2D, threads de 2D, chaque thread calcule 1 element de Aavg
  *
  * Kernel 4: Use 2D blocks and 2D threads, each thread computing 1 element of Aavg, use shared memory
  * Kernel 4: Utiliser blocs de 2D, threads de 2D, , chaque thread calcule 1 element de Aavg, avec shared memory
  *
  * Kernel 5: Use 2D blocks and 2D threads, use shared memory, each thread computes KxK elements of Aavg
  * Kernel 5: Utiliser blocs de 2D, threads de 2D, avec shared memory et KxK ops par thread
  *
  * For all kernels: Make necessary memory allocations/deallocations and memcpy in the main.
  * Pour tous les kernels: Effectuer les allocations/desallocations et memcpy necessaires dans le main.
  */

#include <iostream>
#include <cstdio>
#include <iomanip>
#include "cuda.h"
#include "omp.h"

#define N 16
#define K 4
#define BSXY 16

// The matrix is stored by rows, that is A(i, j) = A[i + j * N]. The average should be computed on Aavg array.
// La matrice A est stockee par lignes, a savoir A(i, j) = A[i + j * N]
float *A;
float *Aavg, *Agpu;
float *dA, *dAavg;

__global__ void computingAverageByBlocks1DNoAdditionalThreads(float *dA, float *dAavg, int n)
{
  // Calculate the start and end indices for the input array
  int start = threadIdx.x * blockDim.x;
  int end = min(threadIdx.x * blockDim.x + blockDim.x, n);

  // Initialize the sum of the 9 points in the current block to 0
  float block_sum = 0.0f;
  // Iterate over the 9 points in the current block and calculate the sum of the points
  for (int i = start; i < end; i++) {block_sum += dA[i];}

  // Calculate the average of the 9 points in the current block
  float block_avg = block_sum / 9.0f;

  // Store the average of the 9 points in the dAavg array
  dAavg[blockIdx.x] = block_avg;
}

__global__ void computingAverageByBlocks2DNoAdditionalThreads(float *dA, float *dAavg, int n)
{
  int i = blockIdx.x;
  int j = blockIdx.y;
  if(i<n-1 && j<n-1 && i != 0 && j!= 0){
      dAavg[i + j * n] = (dA[i - 1 + (j - 1) * n] + dA[i - 1 + (j) * n] + dA[i - 1 + (j + 1) * n] +
          dA[i + (j - 1) * n] + dA[i + (j) * n] + dA[i + (j + 1) * n] +
          dA[i + 1 + (j - 1) * n] + dA[i + 1 + (j) * n] + dA[i + 1 + (j + 1) * n]);
  }
}

__global__ void computingAverageByBlocks2D(float *dA, float *dAavg, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y ;
  if(i<n-1 && j<n-1 && i != 0 && j!= 0)
  {
      dAavg[i + j * N] = (dA[i - 1 + (j - 1) * N] + dA[i - 1 + (j) * N] + dA[i - 1 + (j + 1) * N] +
          dA[i + (j - 1) * N] + dA[i + (j) * N] + dA[i + (j + 1) * N] +
          dA[i + 1 + (j - 1) * N] + dA[i + 1 + (j) * N] + dA[i + 1 + (j + 1) * N]);
  }
}



__global__ void computingAverageByBlocks2DSharedMemory(float *dA, float *dAavg, int n)
{
  __shared__ float shA[BSXY][BSXY];
  int i = blockIdx.x * (BSXY-2) + threadIdx.x;
  int j = blockIdx.y * (BSXY-2) + threadIdx.y;
  shA[threadIdx.x][threadIdx.y] = dA[i + j * N];
  __syncthreads();
    if (threadIdx.x > 0 && i < n - 1 && threadIdx.y > 0 && j < n - 1)
  {
    dAavg[i + j * N] = 
    shA[threadIdx.x - 1][threadIdx.y - 1] + 
    shA[threadIdx.x - 1][threadIdx.y] +
    shA[threadIdx.x - 1][threadIdx.y + 1] + 
    shA[threadIdx.x][threadIdx.y - 1] + 
    shA[threadIdx.x][threadIdx.y] + 
    shA[threadIdx.x][threadIdx.y + 1] +
    shA[threadIdx.x + 1][threadIdx.y - 1] 
    + shA[threadIdx.x + 1][threadIdx.y]+
    shA[threadIdx.x + 1][threadIdx.y + 1];
    }
  __syncthreads();
}

__global__ void computingAverageByBlocks2DSharedMemoryKElements(float *dA, float *dAavg, int n)
{
  __shared__ float shA[BSXY][BSXY];
  int i = blockIdx.x * (BSXY-2) + threadIdx.x;
  int j = blockIdx.y * (BSXY-2) + threadIdx.y;
  shA[threadIdx.x][threadIdx.y] = dA[i + j * N];
  __syncthreads();
    if (threadIdx.x > 0 && i < n - 1 && threadIdx.y > 0 && j < n - 1)
  {
    for (int k = 0; k < K; i++) {
      dAavg[i + j * N] = 
      shA[threadIdx.x - 1][threadIdx.y - 1] + 
      shA[threadIdx.x - 1][threadIdx.y] +
      shA[threadIdx.x - 1][threadIdx.y + 1] + 
      shA[threadIdx.x][threadIdx.y - 1] + 
      shA[threadIdx.x][threadIdx.y] + 
      shA[threadIdx.x][threadIdx.y + 1] +
      shA[threadIdx.x + 1][threadIdx.y - 1] 
      + shA[threadIdx.x + 1][threadIdx.y]+
      shA[threadIdx.x + 1][threadIdx.y + 1];
      }
    }
  __syncthreads();
}


// Reference CPU implementation
// Code de reference pour le CPU
void ninePointAverageCPU(const float *A, float *Aavg)
{
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
          A[i + (j - 1) * N] + A[i + (j) * N] + A[i + (j + 1) * N] +
          A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]);
    }
  }
}

void verifyResults()
{
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      if (Aavg[i + j * N] != Agpu[i + j * N]) {
        std::cout << "incorrect A[" << i << "][" << j << "]" << std::endl;
        return;
      }
    }
  }
  std::cout << "correct!" << std::endl;
}


int main()
{
  A = (float *) malloc (N * N * sizeof(float));
  Aavg = (float *) malloc (N * N * sizeof(float));
  Agpu = (float *) malloc (N * N * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i + j * N] = (float)i * (float)j;
      Agpu[i + j * N] = 0.0f;
    }
  }
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << std::setw(4) << std::setfill(' ') << A[i * N + j];
    }
    std::cout << std::endl;
  }

  ninePointAverageCPU(A, Aavg);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << std::setw(4) << std::setfill(' ') << Aavg[i * N + j];
    }
    std::cout << std::endl;
  }

  cudaMalloc((void **) &dA, N*N*sizeof(float));
  cudaMalloc((void **) &dAavg, N*N*sizeof(float));

  cudaMemcpy(dA, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dAavg, Agpu, N*N*sizeof(float), cudaMemcpyHostToDevice);
  {
    dim3 dimGrid = {N,N};
    dim3 dimBlock = 1;
    computingAverageByBlocks1DNoAdditionalThreads<<<dimGrid, dimBlock>>>(dA, dAavg, N);
    // cudaError_t err = cudaGetLastError();
    // printf("%s\n", cudaGetErrorString(err));
  }


  {
    dim3 dimGrid = {N / BSXY, N / BSXY};
    dim3 dimBlock = 1;
    // computingAverageByBlocks2DNoAdditionalThreads<<<dimGrid, dimBlock>>>(dA, dAavg, N);
  }


  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x = N / BSXY;
    dimGrid.y = N / BSXY;
    dimGrid.z = 1;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    // computingAverageByBlocks2D<<<dimGrid, dimBlock>>>(dA, dAavg, N);
  }

  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x = (N-2) / (BSXY - 2);
    dimGrid.y = (N-2) / (BSXY - 2);
    dimGrid.z = 1;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    computingAverageByBlocks2DSharedMemory<<<dimGrid, dimBlock>>>(dA, dAavg, N);
  }

  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x = N / (BSXY - 2);
    dimGrid.y = N / (BSXY - 2);
    dimGrid.z = 1;
    dimBlock.x = BSXY - 2;
    dimBlock.y = BSXY - 2;
    dimBlock.z = 1;
    // computingAverageByBlocks2DSharedMemoryKElements<<<dimGrid, dimBlock>>>(dA, dAavg, N);
  }

  // cudaMemcpy(dA, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Agpu, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  verifyResults();

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << std::setw(5) << std::setfill(' ') << Aavg[i * N + j];
    }
    std::cout << std::endl;
  }
  printf("\n");
    for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << std::setw(5) << std::setfill(' ') << Agpu[i * N + j];
    }
    std::cout << std::endl;
  }

  free(A);
  free(Aavg);

  return 0;
}