#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // 数组总大小
#define BLOCK_SIZE 256  // 每个 block 中的线程数

// CUDA reduction 内核函数，避免 bank conflict
__global__ void sumReductionOptimized(int *input, int *output, int n) {
    // 使用共享内存，并考虑 bank conflict 的优化
    __shared__ int sharedMem[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int globalIndex = blockIdx.x * blockDim.x * 2 + tid;

    // 读取全局内存数据到共享内存，避免bank冲突
    if (globalIndex < n) {
        sharedMem[tid] = input[globalIndex] + input[globalIndex + blockDim.x];
    } else {
        sharedMem[tid] = 0;  // 超出数组范围的情况处理
    }
    __syncthreads();

    // 执行 sum reduction 操作，步长逐步减小
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();  // 保证所有线程同步
    }

    // 将 block 的结果写入全局内存
    if (tid == 0) {
        output[blockIdx.x] = sharedMem[0];
    }
}

int main() {
    int *h_input, *h_output;
    int *d_input, *d_output;

    // 分配主机和设备内存
    h_input = (int*)malloc(N * sizeof(int));
    h_output = (int*)malloc((N / BLOCK_SIZE) * sizeof(int));

    // 初始化输入数组
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;  // 简单初始化为 1
    }

    // 分配设备内存
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, (N / BLOCK_SIZE) * sizeof(int));

    // 复制数据到设备
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // 启动 CUDA kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sumReductionOptimized<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, N);

    // 复制结果回主机
    cudaMemcpy(h_output, d_output, (N / BLOCK_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);

    // 释放内存
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

