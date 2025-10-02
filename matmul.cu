#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
              cudaGetErrorString(err__), __FILE__, __LINE__);              \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

// Kernel: YOU must implement the math.
// C = A (M x K) * B (K x N)  -> C (M x N)
// Row-major indexing:
//   A[i, k] -> A[i*K + k]
//   B[k, j] -> B[k*N + j]
//   C[i, j] -> C[i*N + j]
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    // TODO:
  }
}

// CPU reference (for correctness check)
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i*K + k] * B[k*N + j];
      }
      C[i*N + j] = sum;
    }
  }
}

// Simple helper to print a matrix
void print_mat(const char* name, const float* X, int R, int C) {
  printf("%s =\n", name);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      printf("%6.1f ", X[i*C + j]);
    }
    printf("\n");
  }
}

int main() {
  // --- Test case ---
  // A(4x3) * B(3x2) = C(4x2)
  const int M = 4, K = 3, N = 2;
  const size_t sizeA = M * K * sizeof(float);
  const size_t sizeB = K * N * sizeof(float);
  const size_t sizeC = M * N * sizeof(float);

  // Host buffers
  float *hA = (float*)malloc(sizeA);
  float *hB = (float*)malloc(sizeB);
  float *hC = (float*)malloc(sizeC);     // GPU result
  float *hC_ref = (float*)malloc(sizeC); // CPU reference

  // Initialize a clear, nontrivial test:
  // A =
  // [1 2 3]
  // [4 5 6]
  // [7 8 9]
  // [1 1 1]
  // B =
  // [1 2]
  // [0 1]
  // [1 0]
  // (Row-major fills below)
  int idx = 0;
  float tempA[12] = {1,2,3, 4,5,6, 7,8,9, 1,1,1};
  for (int i = 0; i < M*K; ++i) hA[i] = tempA[i];

  float tempB[6] = {1,2, 0,1, 1,0};
  for (int i = 0; i < K*N; ++i) hB[i] = tempB[i];

  // Device buffers
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, sizeA));
  CUDA_CHECK(cudaMalloc(&dB, sizeB));
  CUDA_CHECK(cudaMalloc(&dC, sizeC));

  CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, sizeC));

  // Launch config
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y);

  // Launch (will produce wrong result until you implement the kernel)
  matmul_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back
  CUDA_CHECK(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost));

  // CPU reference
  matmul_cpu(hA, hB, hC_ref, M, N, K);

  // Print matrices and results
  print_mat("A (4x3)", hA, M, K);
  print_mat("B (3x2)", hB, K, N);
  print_mat("C_gpu (4x2)", hC, M, N);
  print_mat("C_ref (4x2)", hC_ref, M, N);

  // Check correctness
  int mismatches = 0;
  const float tol = 1e-5f;
  for (int i = 0; i < M*N; ++i) {
    float diff = std::fabs(hC[i] - hC_ref[i]);
    if (diff > tol) mismatches++;
  }

  if (mismatches == 0) {
    printf("\n✅ PASS: GPU matches CPU within tolerance.\n");
  } else {
    printf("\n❌ FAIL: %d mismatches found.\n", mismatches);
  }

  // Cleanup
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  free(hA); free(hB); free(hC); free(hC_ref);

  return 0;
}
