#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

#define N 1024

// Define an static array dA[N] of floats on the GPU
// Definir un tableau de float dA[N] de taille statique sur le GPU
// TODO / A FAIRE ...
float A[N];
__device__ float dA[N];


int main() {
  float A[N], B[N];
  int i;

  // Initialization
  // Initialisation
  for (i = 0; i < N; i++) { A[i] = (float)i; }

  // cudaMemcpy from A[N] to dA[N]
  // cudaMemcpy de A[N] vers dA[N]
  // TODO / A FAIRE ...
  cudaMemcpyToSymbol(dA, A, sizeof(float) * N, 0, cudaMemcpyHostToDevice);
  // cudaMemcpy from dA[N} to B[N]
  // cudaMemcpy de dA[N] vers B[N]
  // TODO / A FAIRE ...
  cudaMemcpyFromSymbol(B, dA, sizeof(float) * N, 0, cudaMemcpyDeviceToHost);

  for (i = 0; i < N; i++) { cout<<B[i];}
  // Wait for GPU kernels to terminate
  // Attendre que les kernels GPUs terminent
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Verify the results
  // Verifier le resultat
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "The copy is incorrect!\n"; }
  else { cout << "The copy is correct!\n"; }

  return 0;
}
