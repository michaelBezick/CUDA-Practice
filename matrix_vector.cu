#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: you need to implement this
__global__ void matvec_kernel(const float *A, const float *x, float *y,
                              int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows) {
    float sum = 0;
    for (int i = 0; i < cols; i++) {
      sum += A[row * cols + i] * x[i];
    }

    y[row] = sum;
  }
}

int main() {
  int rows = 4;
  int cols = 3;

  size_t size_A = rows * cols * sizeof(float);
  size_t size_x = cols * sizeof(float);
  size_t size_y = rows * sizeof(float);

  // Host allocations
  float *h_A = (float *)malloc(size_A);
  float *h_x = (float *)malloc(size_x);
  float *h_y = (float *)malloc(size_y);

  // Initialize A and x with simple values
  for (int i = 0; i < rows * cols; i++) {
    h_A[i] = 1.0f; // all ones
  }
  for (int j = 0; j < cols; j++) {
    h_x[j] = (float)(j + 1); // 1, 2, 3, ...
  }

  // Device allocations
  float *d_A, *d_x, *d_y;
  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_x, size_x);
  cudaMalloc((void **)&d_y, size_y);

  // Copy inputs
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = 128;
  int blocks = (rows + threadsPerBlock - 1) / threadsPerBlock;
  matvec_kernel<<<blocks, threadsPerBlock>>>(d_A, d_x, d_y, rows, cols);

  // Copy result back
  cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);

  // Print result
  printf("Result y:\n");
  for (int i = 0; i < rows; i++) {
    printf("%f\n", h_y[i]);
  }

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);
  free(h_A);
  free(h_x);
  free(h_y);

  return 0;
}
