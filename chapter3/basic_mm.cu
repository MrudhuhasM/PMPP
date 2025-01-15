#include <iostream>
#include <vector>
#include <cuda_runtime.h>

inline void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}


template <typename T>
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
        explicit DeviceBuffer(size_t size_) : d_ptr(nullptr), size(size_){
            if (size > 0){
                checkCuda(cudaMalloc(&d_ptr, size * sizeof(T)), "Allocating device memory");
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

        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (this != &other){
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

        size_t getSize() const {
            return size;
        }
};


__global__ void matMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows_a, int width_a, int width_b){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_a && col < width_b){
        float value = 0.0f;
        for (int k = 0; k < width_a; ++k){
            value += A[row * width_a + k] * B[k * width_b + col];
        }
        C[row * width_b + col] = value;
    }

}

//Exercise Kernel
/*

1.a

Write a kernel that has each thread produce one output matrix row. Fill in
the execution configuration parameters for the design.

*/

__global__ void matMulRowKernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows_a, int width_a, int width_b){

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_a){
        for (int col = 0; col < width_b; ++col){
            float value = 0.0f;
            for (int k = 0; k < width_a; ++k){
                value += A[row * width_a + k] * B[k * width_b + col];
            }
            C[row * width_b + col] = value;
        }
    }

}

/*
1b

Write a kernel that has each thread produce one output matrix column. Fill
in the execution configuration parameters for the design.
*/

__global__ void matMulColKernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows_a, int width_a, int width_b){

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width_b){
        for (int row = 0; row < rows_a; ++row){
            float value = 0.0f;
            for (int k = 0; k < width_a; ++k){
                value += A[row * width_a + k] * B[k * width_b + col];
            }
            C[row * width_b + col] = value;
        }
    }

}



void matrixMultiply(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b, double* elapsedTime){

    size_t size_a = rows_a * cols_a;
    size_t size_b = cols_a * cols_b;
    size_t size_c = rows_a * cols_b;

    DeviceBuffer<float> d_A(size_a);
    DeviceBuffer<float> d_B(size_b);
    DeviceBuffer<float> d_C(size_c);

    checkCuda(cudaMemcpy(d_A.data(), A, size_a * sizeof(float), cudaMemcpyHostToDevice), "Copying A to device");
    checkCuda(cudaMemcpy(d_B.data(), B, size_b * sizeof(float), cudaMemcpyHostToDevice), "Copying B to device");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Creating start event");
    checkCuda(cudaEventCreate(&stop), "Creating stop event");

    dim3 blockSize(16, 16);
    dim3 gridSize((cols_b + blockSize.x - 1) / blockSize.x, (rows_a + blockSize.y - 1) / blockSize.y);

    checkCuda(cudaEventRecord(start), "Recording start event");
    matMulKernel<<<gridSize, blockSize>>>(d_A.data(), d_B.data(), d_C.data(), rows_a, cols_a, cols_b);
    checkCuda(cudaEventRecord(stop), "Recording stop event");
    checkCuda(cudaEventSynchronize(stop), "Synchronizing stop event");

    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");
    *elapsedTime = milliseconds;

    checkCuda(cudaMemcpy(C, d_C.data(), size_c * sizeof(float), cudaMemcpyDeviceToHost), "Copying C to host");   
    
}

void matrixMultiplyRow(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b, double* elapsedTime){

    size_t size_a = rows_a * cols_a;
    size_t size_b = cols_a * cols_b;
    size_t size_c = rows_a * cols_b;

    DeviceBuffer<float> d_A(size_a);
    DeviceBuffer<float> d_B(size_b);
    DeviceBuffer<float> d_C(size_c);

    checkCuda(cudaMemcpy(d_A.data(), A, size_a * sizeof(float), cudaMemcpyHostToDevice), "Copying A to device");
    checkCuda(cudaMemcpy(d_B.data(), B, size_b * sizeof(float), cudaMemcpyHostToDevice), "Copying B to device");

    cudaEvent_t start, stop;

    checkCuda(cudaEventCreate(&start), "Creating start event");
    checkCuda(cudaEventCreate(&stop), "Creating stop event");

    dim3 blockSize(256);
    dim3 gridSize((rows_a + blockSize.x - 1) / blockSize.x);

    checkCuda(cudaEventRecord(start), "Recording start event");
    matMulRowKernel<<<gridSize, blockSize>>>(d_A.data(), d_B.data(), d_C.data(), rows_a, cols_a, cols_b);
    checkCuda(cudaEventRecord(stop), "Recording stop event");
    checkCuda(cudaEventSynchronize(stop), "Synchronizing stop event");

    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");
    *elapsedTime = milliseconds;

    checkCuda(cudaMemcpy(C, d_C.data(), size_c * sizeof(float), cudaMemcpyDeviceToHost), "Copying C to host");   
    
}


void matrixMultiplyCol(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b, double* elapsedTime){

    size_t size_a = rows_a * cols_a;
    size_t size_b = cols_a * cols_b;
    size_t size_c = rows_a * cols_b;

    DeviceBuffer<float> d_A(size_a);
    DeviceBuffer<float> d_B(size_b);
    DeviceBuffer<float> d_C(size_c);

    cudaEvent_t start, stop;

    checkCuda(cudaEventCreate(&start), "Creating start event");
    checkCuda(cudaEventCreate(&stop), "Creating stop event");

    checkCuda(cudaMemcpy(d_A.data(), A, size_a * sizeof(float), cudaMemcpyHostToDevice), "Copying A to device");
    checkCuda(cudaMemcpy(d_B.data(), B, size_b * sizeof(float), cudaMemcpyHostToDevice), "Copying B to device");

    dim3 blockSize(256);
    dim3 gridSize((cols_b + blockSize.x - 1) / blockSize.x);

    checkCuda(cudaEventRecord(start), "Recording start event");
    matMulColKernel<<<gridSize, blockSize>>>(d_A.data(), d_B.data(), d_C.data(), rows_a, cols_a, cols_b);
    checkCuda(cudaEventRecord(stop), "Recording stop event");
    checkCuda(cudaEventSynchronize(stop), "Synchronizing stop event");

    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");
    *elapsedTime = milliseconds;

    checkCuda(cudaMemcpy(C, d_C.data(), size_c * sizeof(float), cudaMemcpyDeviceToHost), "Copying C to host");   
    
}

double benchmarkMatrixMultiply(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b, int iterations){

    double totalTime = 0.0;
    double elapsedTime = 0.0;

    //warm-up
    double warmupTime = 0.0;
    matrixMultiply(A, B, C, rows_a, cols_a, cols_b, &warmupTime);

    for (int i = 0; i < iterations; ++i){
        matrixMultiply(A, B, C, rows_a, cols_a, cols_b, &elapsedTime);
        totalTime += elapsedTime;
    }

    return totalTime / iterations;
}

double benchmarkMatrixMultiplyRow(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b, int iterations){

    double totalTime = 0.0;
    double elapsedTime = 0.0;

    //warm-up
    double warmupTime = 0.0;
    matrixMultiplyRow(A, B, C, rows_a, cols_a, cols_b, &warmupTime);

    for (int i = 0; i < iterations; ++i){
        matrixMultiplyRow(A, B, C, rows_a, cols_a, cols_b, &elapsedTime);
        totalTime += elapsedTime;
    }

    return totalTime / iterations;
}

double benchmarkMatrixMultiplyCol(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b, int iterations){

    double totalTime = 0.0;
    double elapsedTime = 0.0;

    //warm-up
    double warmupTime = 0.0;
    matrixMultiplyCol(A, B, C, rows_a, cols_a, cols_b, &warmupTime);

    for (int i = 0; i < iterations; ++i){
        matrixMultiplyCol(A, B, C, rows_a, cols_a, cols_b, &elapsedTime);
        totalTime += elapsedTime;
    }

    return totalTime / iterations;
}


int main(){
    /*    
    A = 1920 x 1330
    B = 1330 x 1920
    C = 1920 x 1920    
    */

    int rows_a = 1920;
    int cols_a = 1330;

    int rows_b = 1330;
    int cols_b = 1920;

    int rows_c = 1920;
    int cols_c = 1920;

    size_t size_a = rows_a * cols_a;
    size_t size_b = rows_b * cols_b;
    size_t size_c = rows_c * cols_c;

    std::vector<float> h_a(size_a, 1.0f);
    std::vector<float> h_b(size_b, 2.0f);
    std::vector<float> h_c(size_c, 0.0f);


    double dummyTime = 0.0;
    matrixMultiply(h_a.data(), h_b.data(), h_c.data(), rows_a, cols_a, cols_b, &dummyTime);

    // Verify result
    for (size_t i = 0; i < size_c; ++i){
        if (h_c[i] != cols_a * 2.0f){
            std::cerr << "Mismatch at index " << i << ": " << h_c[i] << std::endl;
            return -1;
        }
    }

    std::cout << "Benchmarking Matrix Multiplication naive kernel..." << std::endl;
    std::cout << "===========================================" << std::endl;
    auto avgTime = benchmarkMatrixMultiply(h_a.data(), h_b.data(), h_c.data(), rows_a, cols_a, cols_b, 10);
    std::cout << "Average time over 10 iterations: " << avgTime << " ms" << std::endl;    
    double gflops = (2.0 * rows_a * cols_a * cols_b) / (avgTime * 1e6);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "===========================================" << std::endl;

    std::cout << "Benchmarking Matrix Multiplication Row kernel... Exercise 1.a" << std::endl;
    std::cout << "===========================================" << std::endl;
    auto avgTimeRow = benchmarkMatrixMultiplyRow(h_a.data(), h_b.data(), h_c.data(), rows_a, cols_a, cols_b, 10);
    std::cout << "Average time (Row Kernel) over 10 iterations: " << avgTimeRow << " ms" << std::endl;
    double gflopsRow = (2.0 * rows_a * cols_a * cols_b) / (avgTimeRow * 1e6);
    std::cout << "Performance (Row Kernel): " << gflopsRow << " GFLOPS" << std::endl;
    std::cout << "===========================================" << std::endl;


    std::cout << "Benchmarking Matrix Multiplication Column kernel... Exercise 1.b" << std::endl;
    std::cout << "===========================================" << std::endl;
    auto avgTimeCol = benchmarkMatrixMultiplyCol(h_a.data(), h_b.data(), h_c.data(), rows_a, cols_a, cols_b, 10);
    std::cout << "Average time (Column Kernel) over 10 iterations: " << avgTimeCol << " ms" << std::endl;
    double gflopsCol = (2.0 * rows_a * cols_a * cols_b) / (avgTimeCol * 1e6);
    std::cout << "Performance (Column Kernel): " << gflopsCol << " GFLOPS" << std::endl;
    std::cout << "===========================================" << std::endl;

    std::cout << "Matrix multiplication successful!" << std::endl;
    return 0;

}