#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel function to compute vector addition
__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    // Calculate the global index for the thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Perform the addition only if the index is within bounds
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Host function to initialize vectors and launch the CUDA kernel
int main()
{
    // Size of vectors
    int n = 1000000;
    
    // Size in bytes for each vector
    size_t bytes = n * sizeof(float);
    
    // Allocate memory for vectors on host
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize vectors A and B with some values
    for(int i = 0; i < n; i++) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 0.5f;
    }

    // Allocate memory for vectors on device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Number of threads in a block
    int blockSize = 256;
    
    // Number of blocks in the grid
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch the kernel on the device
    vecAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    
    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy the result vector from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify the result (optional)
    bool correct = true;
    for(int i = 0; i < n; i++) {
        if(h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            break;
        }
    }
    if(correct)
        std::cout << "Vector addition is correct!" << std::endl;
    else
        std::cout << "Error in vector addition!" << std::endl;

    // Free memory on device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free memory on host
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
