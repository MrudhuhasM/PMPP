#include <iostream>
#include <vector>
#include <cuda_runtime.h>


inline void checkCudaError(cudaError_t err, const char* msg){
    if(err != cudaSuccess){
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


template<typename T>
class DeviceBuffer{
    private:
        T* d_ptr;
        size_t size;

        void release(){
            if(d_ptr){
                cudaFree(d_ptr);
                d_ptr = nullptr;
            }
        }
    public:
        DeviceBuffer() : d_ptr(nullptr), size(0) {}
        explicit DeviceBuffer(size_t n) : d_ptr(nullptr), size(n){
            if (n > 0){
                checkCudaError(cudaMalloc(&d_ptr, n * sizeof(T)), "Allocating device memory");
            }
        }
        ~DeviceBuffer(){
            release();
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        DeviceBuffer(DeviceBuffer&& other) noexcept : d_ptr(other.d_ptr), size(other.size){
            other.d_ptr = nullptr;
            other.size = 0;
        }

        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept{
            if(this != &other){
                release();
                d_ptr = other.d_ptr;
                size = other.size;
                other.d_ptr = nullptr;
                other.size = 0;
            }
            return *this;
        }

        T* data(){
            return d_ptr;
        }

        const T* data() const {
            return d_ptr;
        }

        size_t getSize() const {
            return size;
        }
        
};


__global__ void vecAddKernel(const int* __restrict__ A, const int* __restrict__ B, int* C, size_t N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

void vecAdd(const int* A, const int* B, int* C, size_t N, double* kernelTimeMs){
    DeviceBuffer<int> d_A(N);
    DeviceBuffer<int> d_B(N);
    DeviceBuffer<int> d_C(N);
    checkCudaError(cudaMemcpy(d_A.data(), A, N * sizeof(int), cudaMemcpyHostToDevice), "Copying A to device");
    checkCudaError(cudaMemcpy(d_B.data(), B, N * sizeof(int), cudaMemcpyHostToDevice), "Copying B to device");
    

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Timing setup
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Creating start event");
    checkCudaError(cudaEventCreate(&stop), "Creating stop event");
    
    checkCudaError(cudaEventRecord(start), "Recording start event");
    vecAddKernel<<<numBlocks, blockSize>>>(d_A.data(), d_B.data(), d_C.data(), N);
    checkCudaError(cudaEventRecord(stop), "Recording stop event");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop event");
    
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");
    *kernelTimeMs = milliseconds;
    
    checkCudaError(cudaEventDestroy(start), "Destroying start event");
    checkCudaError(cudaEventDestroy(stop), "Destroying stop event");
    
    checkCudaError(cudaGetLastError(), "Launching vecAddKernel");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after kernel launch");
    checkCudaError(cudaMemcpy(C, d_C.data(), N * sizeof(int), cudaMemcpyDeviceToHost), "Copying C to host");
}

double benchmarkVecAdd(const int* A, const int* B, int* C, size_t N, int iterations){

    // Warmup
    double dummyTime;
    vecAdd(A, B, C, N, &dummyTime);
    
    double totalTime = 0.0;
    for (int i = 0; i < iterations; ++i) {
        double kernelTime;
        vecAdd(A, B, C, N, &kernelTime);
        totalTime += kernelTime;
    }
    return totalTime / iterations;
}



int main(){
    size_t N = 1 << 20; 
    std::vector<int> A_h(N, 1);
    std::vector<int> B_h(N, 2);
    std::vector<int> C_h(N, 0);

    double dummyTime;
    vecAdd(A_h.data(), B_h.data(), C_h.data(), N, &dummyTime);

    
    for(size_t i = 0; i < N; ++i){
        if(C_h[i] != 3){
            std::cerr << "Verification failed at index " << i << ": " << C_h[i] << " != 3" << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Verification passed!" << std::endl;

    std::cout << "Benchmarking vecAdd..." << std::endl;
    int iterations = 10;

    double avgKernelTime = benchmarkVecAdd(A_h.data(), B_h.data(), C_h.data(), N, iterations);
    std::cout << "Average kernel time per vecAdd: " << avgKernelTime << " ms" << std::endl;

    // Calculate GIOPS (1 operation per element: the add)
    double giops = (1.0 * N) / (avgKernelTime / 1000.0) / 1e9;
    std::cout << "Performance: " << giops << " GIOPS" << std::endl;

    return EXIT_SUCCESS;
}