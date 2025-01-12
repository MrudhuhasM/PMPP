#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call) do {                              \
    cudaError_t err = call;                                \
    if(err != cudaSuccess){                                \
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n",   \
                cudaGetErrorString(err), err);            \
        exit(err);                                         \
    }                                                      \
} while (0)

#define CHECK_CUDA_LAUNCH() do {                            \
    cudaError_t err = cudaGetLastError();                \
    if(err != cudaSuccess){                              \
        fprintf(stderr, "CUDA Launch Error: %s (err_num=%d)\n", \
                cudaGetErrorString(err), err);          \
        exit(err);                                       \
    }                                                    \
} while (0)


__global__ void vector_add_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size){
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalThreadId < size){
        C[globalThreadId] = A[globalThreadId] + B[globalThreadId];
    }
}


void vec_add_cpu(const float* __restrict__ A_h, const float* __restrict__ B_h, float* __restrict__ C_h, int size){
    for(int i=0; i<size; i++){
        C_h[i] = A_h[i] + B_h[i];
    }
}



void vec_add(const float* __restrict__ A_h, const float* __restrict__ B_h, float* __restrict__ C_h, int size){
    
    size_t size_in_bytes = static_cast<size_t>(size) * sizeof(float);
    float *A_d, *B_d, *C_d;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&A_d), size_in_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&B_d), size_in_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&C_d), size_in_bytes));

    CHECK_CUDA(cudaMemcpy(A_d, A_h, size_in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h, size_in_bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(A_d, B_d, C_d, size);
    CHECK_CUDA_LAUNCH();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float bandwidth = (3.0f * size_in_bytes) / (milliseconds * 1e6f); // GB/s
    std::cout << "GPU Time: " << milliseconds << "ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaMemcpy(C_h, C_d, size_in_bytes, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
}

int main(){
    const int size = 1<<25;
    std::vector<float> A_h(size), B_h(size), C_h_GPU(size), C_h_CPU(size);

    for(int i=0; i<size; i++){
        A_h[i] = i * 1.0f;
        B_h[i] = i * 2.0f;
    }

    vec_add(A_h.data(), B_h.data(), C_h_GPU.data(), size);

    auto start = std::chrono::high_resolution_clock::now();
    vec_add_cpu(A_h.data(), B_h.data(), C_h_CPU.data(), size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end - start;
    std::cout << "CPU Time: " << cpu_duration.count() << "ms" << std::endl;

    for(int i=0; i<size; i++){
        if(fabs(C_h_GPU[i] - C_h_CPU[i]) > 1e-5){
            std::cerr << "Mismatch at index " << i << ": GPU " << C_h_GPU[i] << " != CPU " << C_h_CPU[i] << std::endl;
            return -1;
        }
    }
    std::cout << "Results match!" << std::endl;
    return 0;

}