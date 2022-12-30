#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <iomanip>

int N = 1024;
const int nStreams = 4;
float *A, *B, *C;
float *dA, *dB, *dC;
cudaStream_t streams[nStreams];

// Kernel that performs the matrix vector multiplication b(i) = sum_j(A(i, j), x(j))
// A is row-major (stored row-by-row in memory)
__global__ void matvec(float *dA, float *x, float *b, int n)
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float res = 0;
  for (int j = 0; j < n; j++) { res += dA[i * n + j] * x[j]; }
  b[i] = res;
}


int main()
{
  // A is stored by rows, A(i, j) = A[i * N + j]
  A = (float *) malloc (N * N * sizeof(float));
  // B and C are stored by columns, B(i, j) = B[i + j * N]
  B = (float *) malloc (N * N * sizeof(float));
  C = (float *) malloc (N * N * sizeof(float));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i + j; // A(i, j) = i + j
      B[i * N + j] = i - j; // B(j, i) = i - j
      C[i * N + j] = 0; // C(j, i) = 0
    }
  }
  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dB, N * nStreams * sizeof(float));
  cudaMalloc(&dC, N * nStreams * sizeof(float));

  // Only copy the entire matrix A. For B and C, they need to be copied and computed one column vector at a time in a streaming manner
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Create streams
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }
  for (int j = 0; j < N; j++) {
    // Copy the column j of B into one of slots in dB using the stream no (j % nStreams) and cudaMemcpyAsync
    cudaMemcpyAsync(dB + (j % nStreams) * N, B + j * N, N * sizeof(float), cudaMemcpyHostToDevice, streams[j % nStreams]);

    // Perform the matrix-vector multiplication on A and the column vector in dB(:, j % nStreams), compute on dC(:, j % nStreams), using stream no (j % nStreams)
    int blockSize = 32;
    int nBlocks = (N + blockSize - 1) / blockSize;
    matvec<<<nBlocks, blockSize, 0, streams[j % nStreams]>>>(dA, dB + (j % nStreams) * N, dC + (j % nStreams) * N, N);
    
    // Copy back the computed vector dC(:, j % nStreams) into the column C(:, j) using the same stream no (j % nStreams) and cudaMemcpyAsync
    cudaMemcpyAsync(C + j * N, dC + (j % nStreams) * N, N * sizeof(float), cudaMemcpyDeviceToHost, streams[j % nStreams]);
  }

  cudaDeviceSynchronize();

  free(A); free(B); free(C);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
