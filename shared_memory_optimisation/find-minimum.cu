#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "cuda.h"
#include <cfloat>
#include <iomanip>
/** It did NOT run properly, but I have really tried, sorry :( 
  I understand the process though~ **/
#define BLOCKSIZE 1024


/**
  * Version 1: Write a 1D GPU kernel that finds the minimum element of an array dA[N] for each block and writes the minimum of each block to a bin of dAmin. Then, CPU takes dAmin and calculates the global minimum in sequence on this small table.
   *
   * Version 2: The first call to findMinimum reduces the size of the array to be scanned sequentially to N/BLOCKSIZE. In this version, use findMinimum twice in a row to reduce the size of the array to be scanned sequentially to N/(BLOCKSIZE*BLOCKSIZE) (so that the sequential part in CPU becomes really negligible).
   *
   * To find the minimum of the two floats in GPU, use the function fminf(x, y)

  */

__global__ void findMinimum(float *dA, float *dAmin, int N) {
  // Calculate the starting index for the current block
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Declare the shared memory array
  __shared__ volatile float buff[BLOCKSIZE];

  // Initialize the minimum value for the current block to the
  // maximum possible floating-point value
  float blockMin = FLT_MAX;
  for (int i; i < N; i += BLOCKSIZE) {
    buff[threadIdx.x] = dA[i];
    __syncthreads();
    for (int j = 0; j < BLOCKSIZE; j++) {
      blockMin = fminf(blockMin, buff[j]);
    }
    __syncthreads();
  }

  // Write the minimum value for the current block to the
  // corresponding bin in dAmin
  dAmin[blockIdx.x] = blockMin;
}

using namespace std;

int main()
{
  srand(1234);
  int N = 100000000;
  int numBlocks = N/BLOCKSIZE + 1;// = ???; (A FAIRE ...)
  float *A, *dA; // Le tableau dont minimum on va chercher
  float *Amin, *dAmin; // Amin contiendra en suite le tableau reduit par un facteur de BLOCKSIZE apres l'execution du kernel GPU

  // Allocate the arrays A[N] and Amin[numBlocks] in a ``pined'' way on the CPU
  // Allocate the dA[N] and dAmin[numBlocks] arrays on the GPU
  A = (float *) malloc (N * sizeof(float));
  Amin = (float *) malloc (numBlocks * sizeof(float));
  cudaMalloc((void **) &dA, N*sizeof(float));
  cudaMalloc((void **) &dAmin, numBlocks*sizeof(float));

  // Initialiser le tableau A
  for (int i = 0; i < N; i++) { A[i] = (float)(rand() % 1000); }
  A[rand() % N] = -1.0; // Mettre le minimum a -1.

  // Put A on the GPU (dA) with memcpy
  cudaMemcpy(dA, A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dAmin, Amin, numBlocks*sizeof(float), cudaMemcpyHostToDevice);

  cudaError_t err = cudaGetLastError();
  printf("%s\n", cudaGetErrorString(err));
  float minA = FLT_MAX; // Affecter le maximum float a minA
  // Find the minimum of the dE array, put admin in the CPU, then find the global minimum and put it in the variable minA
  //version 1
  {
    dim3 dimGrid = numBlocks;
    dimGrid.x = numBlocks;
    dimGrid.y = numBlocks;
    dim3 dimBlock = BLOCKSIZE;
    findMinimum<<<dimGrid, dimBlock>>>(dA, dAmin, N);
  }

    // version 2
  //   {
  //   dim3 dimGrid = numBlocks;
  //   dimGrid.x = numBlocks;
  //   dimGrid.y = numBlocks;
  //   dim3 dimBlock = BLOCKSIZE;
  //   findMinimum<<<dimGrid, dimBlock>>>(dA, dAmin, N);
  //   cudaMemcpy(Amin, dAmin, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

  //   int new_N = numBlocks;
  //   int new_numBlocks = new_N/BLOCKSIZE + 1;  
  //   dim3 new_dimGrid = new_numBlocks;
  //   dimGrid.x = new_numBlocks;
  //   dimGrid.y = new_numBlocks;
  //   dim3 new_dimBlock = BLOCKSIZE;
  //   // Put A on the GPU (dA) with memcpy
  //   cudaMemcpy(dAmin, Amin, new_N*sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(dAmin, Amin, new_numBlocks*sizeof(float), cudaMemcpyHostToDevice);
  //   findMinimum<<<new_dimGrid, new_dimBlock>>>(Amin, dAmin, N);
  // }

  cudaMemcpy(Amin, dAmin, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

  // find the global minimum and put it in the variable minA
  int min_value = Amin[0];
  for (int i = 1; i < Amin.size(); i++) {
    if (Amin[i] < min_value) {
        minA = Amin[i];
    }
  }

  // Verifier le resultat
  if (minA == -1) { cout << "The minimum is correct!" << endl; }
  else { cout << "The minimum found (" << minA << ") is incorrect (it should have been -1)!" << endl; }
  // Deallocate dA and dAt
  cudaFree(dA); cudaFree(dAmin); 

  // Deallocate A and At
  free(A);
  free(Amin);
  return 0;
}
