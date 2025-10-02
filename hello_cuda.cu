#include <stdio.h>

// A CUDA kernel is marked with __global__
// This runs on the GPU, callable from the CPU
__global__ void hello_cuda() {
    printf("Hello from GPU thread (%d,%d)!\n", blockIdx.x, threadIdx.x);
}

int main() {
    // Launch kernel with <<<grid, block>>>
    // Here: 2 blocks, each with 4 threads
    hello_cuda<<<2, 4>>>();

    // Synchronize to wait for GPU to finish
    cudaDeviceSynchronize();

    printf("Hello from CPU!\n");
    return 0;
}
