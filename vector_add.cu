#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  int N = 1 << 16; // 65536 elements
  size_t size = N * sizeof(float);

  // Allocate host memory
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Copy data to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Launch kernel with 256 threads per block
  int threadsPerBlock = 256;
  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  vector_add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

  // Copy result back
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // Verify result
  for (int i = 0; i < 5; i++) {
    printf("h_c[%d] = %f\n", i, h_c[i]);
  }

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
