#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLK 128

// 定义cuda卷积核函数
__global__ void convolve(int N, float* input, float* output) {
    __shared__ float support[THREADS_PER_BLK + 2];      // 每一个block一个shared mem

    int index = blockIdx.x * blockDim.x + threadIdx.x;  // 通过block index和thread index定位thread
    if (index < N) {
        support[threadIdx.x] = input[index];
        if (threadIdx.x < 2 && index + THREADS_PER_BLK < N) {
                support[THREADS_PER_BLK + threadIdx.x] = input[index + THREADS_PER_BLK];
        }

        // 类似于线程中的barrier
        __syncthreads();

        float result = 0.0f;    
        for (int i = 0; i < 3; i++) {
            result += support[threadIdx.x + i];
        }
        output[index] = result / 3.f;
    }
}

// main 函数
int main() {
    const int N = 1024 * 1024;
    size_t size = N * sizeof(float);

    // Host memory
    float* h_input = new float[N];
    float* h_output = new float[N];

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Device memory
    float *d_input, *d_output;
    // 开辟 device global memory
    cudaMalloc(&d_input, size + 2 * sizeof(float));  // Allocate extra space for boundary conditions
    cudaMalloc(&d_output, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    // Launch kernel
    convolve<<<(N + THREADS_PER_BLK - 1) / THREADS_PER_BLK, THREADS_PER_BLK>>>(N, d_input, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print some results for verification
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    delete[] h_input;
    delete[] h_output;

    return 0;
}
