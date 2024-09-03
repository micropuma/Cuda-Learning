#include <iostream>

// CUDA kernel 函数，负责在 GPU 上并行运行
__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x;  // 获取当前线程的索引
    if (index < n) {
        c[index] = a[index] + b[index];  // 执行加法操作
    }
}

int main() {
    const int N = 5;
    int h_a[N], h_b[N], h_c[N];  // 主机（Host）上的数组
    int *d_a, *d_b, *d_c;  // 设备（Device）上的数组指针

    // 分配设备内存
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // 初始化主机上的数组
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 将数据从主机传输到设备
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // 在 GPU 上执行 kernel 函数，N 个线程
    add<<<1, N>>>(d_a, d_b, d_c, N);

    // 将结果从设备传回主机
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 输出结果
    std::cout << "Result: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
