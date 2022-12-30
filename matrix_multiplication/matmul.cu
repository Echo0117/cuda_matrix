#include <cstdio>  
#include <iostream>
#include "cuda.h"  

using namespace std;

#define N 128

// A and C are stored by rows, i.e., A(i, j) = A[i * N + j], C(i, j) = C[i * N + j]
// B is stored by columns, i.e., B(i, j) = B[i + j * N]
// A et C sont stockes par lignes, a savoir A(i, j) = A[i * N + j], C(i, j) = C[i * N + j]
// B est stocke par colonne, a savoir B(i, j) = B[i + j * N]
float *A, *B, *C;

// dA and dC are stored by rows, dC is stored by columns
// dA et dC sont stockes par lignes, dC est stocke par colonne
float *dA, *dB, *dC;


// Create a block for computing each element C(i, j), compute using 1 thread by block
// Creer un bloc pour le calcul de chaque element C(i, j), calculer avec 1 thread par bloc
__global__ void multiplyMatrixGPUByBlocks(float *dA, float *dB, float *dC, int n)
{
  // TODO / A FAIRE ...
  int i = blockIdx.x;
  int j = blockIdx.y;

  for (int k = 0; k < n; k++) { 
    dC[i * N + j] += dA[i * N + k] * dB[k + j * N];
  }
}


// Create a block for computing blockDim.x elements of C, compute using blockDim.x threads per block. Each thread computes one element of C
// Assume N is a multiple of blockDim.x
// Creer un bloc pour le calcul de blockDim.x elements de C, calculer avec blockDim.x threads par bloc. Chaque thread calcule un element de C.
// Supposer que N est un divisible par blockDim.x
__global__ void multiplyMatrixGPUByBlocksThreads1D(float *dA, float *dB, float *dC, int n)
{
  // TODO / A FAIRE ...
  int i = blockIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  for (int k = 0; k < n; k++) { 
    dC[i * N + j] += dA[i * N + k] * dB[k + j * N];
  }
}


// Create a block for computing blockDim.x elements of C, compute using blockDim.x threads per block. Each thread computes one element of C
// Make it work when N is not a multiple of blockDim.x
// Creer un bloc pour le calcul de blockDim.x elements de C, calculer avec blockDim.x threads par bloc.
// Chaque thread calcule un element de C.
// Faire marcher pour N n'est pas multiple de blockDim.x.
__global__ void multiplyMatrixGPUByBlocksThreads1DNonMultiple(float *dA, float *dB, float *dC, int n)
{
  // TODO / A FAIRE ...
  int i = blockIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  for (int k = 0; k < n; k++) { 
    if (j < n)
      {
        dC[i * N + j] += dA[i * N + k] * dB[k + j * N];
      }
  }
}



// Create a block for computing blockDim.x * blockDim.y elements of C, compute using blockDim.x * blockDim.y threads per block.
// Each thread computes one element of C.
// Assume N is a multiple of blockDim.x
// Creer un bloc pour le calcul de blockDim.x * blockDim.y elements de C, calculer avec blockDim.x * blockDim.y threads par bloc.
// Chaque thread calcule un element de C.
// Supposer que N est un divisible par blockDim.x
__global__ void multiplyMatrixGPUByBlocksThreads2D(float *dA, float *dB, float *dC, int n)
{
  // TODO / A FAIRE ...
  // printf("A = %d\n",gridDim);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  for (int k = 0; k < n; k++) { 
    dC[i * N + j] += dA[i * N + k] * dB[k + j * N];
  }
}


// Create a block for computing blockDim.x * blockDim.y elements of C, compute using blockDim.x * blockDim.y threads per block. Each thread computes one element of C
// Make it work when N is not a multiple of blockDim.x nor blockDim.y
// Creer un bloc pour le calcul de blockDim.x * blockDim.y elements de C, calculer avec blockDim.x * blockDim.y threads par bloc.
// Chaque thread calcule un element de C.
// Faire marcher pour N n'est pas multiple de ni blockDim.x ni blockDim.y.
__global__ void multiplyMatrixGPUByBlocksThreads2DNonMultiple(float *dA, float *dB, float *dC, int n)
{
  // TODO / A FAIRE ...
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  for (int k = 0; k < n; k++) 
    if (j < n && i < n){
      { 
        dC[i * N + j] += dA[i * N + k] * dB[k + j * N];
      }
    }
}


// Reference CPU code for multipying matrices C = AB (A, C stored by rows, B stored by columns)
// Code reference de CPU pour effectuer la multiplication de matrices C = AB (A, C stockes par ligne, B stocke par colonne)
void multiplyMatrixCPU()
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0.0f;
      for (int k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k + j * N];
      }
    }
  }
}

void verifyResults()
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float c = 0.0f;
      for (int k = 0; k < N; k++) {
        c += A[i * N + k] * B[k + j * N];
        // printf("heya %f\n", c);
      }

      if (std::abs(C[i * N + j] - c) > 1e-6) {
        std::cout << "Multiplication is incorrect for the element C[" << i << "][" << j << "]" << std::endl;
        return;
      }
    }
  }
  std::cout << "Multiplication is correct!" << std::endl;
}

int main(int argc, char **argv)
{
  // Initialization
  // Initialisation
  A = (float *)malloc(N * N * sizeof(A[0]));
  B = (float *)malloc(N * N * sizeof(B[0]));
  C = (float *)malloc(N * N * sizeof(C[0]));
  for (int j = 0; j < N; j++) { 
    for (int i = 0; i < N; i++) { 
      A[i + j * N] = i + j; // A(i, j) = i + j
      B[i + j * N] = 1.0f; // B(j, i) = 1
      C[i + j * N] = 0.0f;
    }
  }
  
  // Allocate dA and dB, then copy the arrays A and B to the GPU
  // Allouer dA et dB, puis copier les tableaux A et B vers le GPU
  // TODO / A FAIRE ...
  cudaMalloc((void **) &dA, N*N*sizeof(float));
  cudaMalloc((void **) &dB, N*N*sizeof(float));
  cudaMalloc((void **) &dC, N*N*sizeof(float));


  cudaMemcpy(dA, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, N*N*sizeof(float), cudaMemcpyHostToDevice);

  // Call each GPU kernel appropriately to multiply matrices A and B
  // Measure and print the execution time and performance (GFlops/s) of each kernel, without counting the data transfer time
  // Appeler chaque kernel GPU de maniere appropriee pour multiplier les matrices A et B
  // Mesurer et afficher le temps d'execution et la performance (en GFlops/s) de chaque kernel, sans compter le temps de transfert.
  // TODO / A FAIRE ...


  // I commented each functions as follows as the inputs dA, dB, dC are changing, 
  // you need to uncomment them for testing different functions :)
  {
    dim3 dimGrid = {N, N};
    dim3 dimBlock = 3;
    multiplyMatrixGPUByBlocks<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  }

  {
    dim3 dimGrid{N, N};
    dim3 dimBlock = 2;
    // multiplyMatrixGPUByBlocksThreads1D<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  }
  { 
    dim3 dimGrid = {N, N};
    dim3 dimBlock = 2;
    // multiplyMatrixGPUByBlocksThreads1DNonMultiple<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  }
  {
    dim3 dimGrid(4, 4);
    dim3 dimBlock(32, 32);
    // multiplyMatrixGPUByBlocksThreads2D<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  } 
  {
    dim3 dimGrid = {N, N};
    dim3 dimBlock(4, 4);
    // multiplyMatrixGPUByBlocksThreads2DNonMultiple<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  }

  // Copy the array dC back to the CPU
  // Recopier le tableau dC vers le CPU
  // TODO / A FAIRE ...
  cudaMemcpy(C, dC, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  // Verify the results
  // Verifier les resultats
  // multiplyMatrixCPU();
  verifyResults();

  // Deallocate A, B, C
  // Desallouer A, B, C
  free(A); free(B); free(C);

  // Deallocate dA, dB, dC
  // Desallouer dA, dB, dC
  // TODO / A FAIRE ...
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
