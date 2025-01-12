Notes from 2nd Chapter of "Programming Massively Parallel Processors: A Hands-on Approach" 

# Vector Addition

## Discussed Data Parallelism
Data parallelism means doing the same operation on different pieces of data at the same time.

In CUDA, this maps perfectly to how GPUs are built:

- we have thousands of lightweight threads,
- grouped into warps (32 threads),
- which execute the same instruction (like C[i] = A[i] + B[i]) but on different data indices.

We have Thread for single operation, this executes kernel code on one data element.
We have Block for group of threads, this executes kernel code on a chunk of data elements.
We have Grid for group of blocks, this executes kernel code on the entire data set.

GPUs are built around SIMT (Single Instruction, Multiple Threads) architecture. Which takes advantage of data parallelism by executing the same instruction across multiple threads simultaneously.

There is also Task Parallelism, which is about doing different operations on the same or different data at the same time. This is less common in CUDA programming.

From Book:

> In general, data parallelism is the main source of scalability for parallel programs. With large datasets, one can often find abundant data parallelism to be able to utilize massively parallel processors and allow application performance to grow with each generation of hardware that has more execution resources. Nevertheless, task parallelism can also play an important role in achieving performance goals. We will be covering task parallelism later when we introduce streams.

## CUDA C Programming Model
CUDA C extends C/C++ by allowing the programmer to define functions, called kernels, that are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C functions. The programmer specifies the number of threads to be used when launching a kernel.

Structure of CUDA C reflects the coexistence host (CPU) and device (GPU) code. 

The execution starts with host code, when a kernel function is called large number of threads are launched in parallel on the device to execute the kernel code.All threads launched by kernel are called a grid. These grid of threads exploit data parallelism. We can assume these threads takes very few clock cycles to generate and schedule in contrast to traditional CPU threads.

> A thread is a simplified view of how a processor executes a sequential program in modern computers. A thread consists of the code of the program,the point in the code that is being executed, and the values of its variablesand data structures. The execution of a thread is sequential as far as auser is concerned. One can use a source-level debugger to monitor the progress of a thread by executing one statement at a time, looking at the statement that will be executed next and checking the values of the variables and data structures as the execution progresses.Threads have been used in programming for many years. If a programmer wants to start parallel execution in an application, he/she creates and manages multiple threads using thread libraries or special languages. In CUDA, the execution of each thread is sequential as well. A CUDA program initiates parallel execution by calling kernel functions, which causes the underlying runtime mechanisms to launch a grid of threads that process different parts of the data in paralle

Current CUDA system and devices come with Device Global Memory, before starting kernel we need to allocate memory on device and copy data from host to device. After kernel execution we need to copy results back from device to host.



Built in Variables:
- `threadIdx`: Unique thread index within a block.
- `blockIdx`: Unique block index within a grid.
- `blockDim`: Number of threads in a block.
- `gridDim`: Number of blocks in a grid.
- `__global__`: Qualifier to declare a kernel function that runs on the device and is called from the host.
- `__device__`: Qualifier to declare a function that runs on the device and is called from other device functions or kernels.
- `__host__`: Qualifier to declare a function that runs on the host and is called from the host (default for regular C/C++ functions).


Each thread has a unique ID, which can be calculated using `threadIdx`, `blockIdx`, and `blockDim`. This ID is used to determine which part of the data the thread will operate on.
The unique global thread ID can be calculated as:
```c
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
```

Kernel Launch Syntax:
```c    
kernelFunction<<<numBlocks, threadsPerBlock>>>(arguments);
```

Where:
- `numBlocks`: Number of blocks in the grid.
- `threadsPerBlock`: Number of threads in each block.
- `<<< >>>`: Special syntax to launch a kernel with specified grid and block dimensions.


Functions and Keywords from this Chapter and Code Example:

- `cudaMalloc()`: 
    - Allocates memory on the device.
    - 2 arguments: pointer to pointer and size in bytes.
- `cudaFree()`: Frees memory on the device.
- `cudaMemcpy()`:
    - Copies data between host and device.
    - 4 arguments: destination pointer, source pointer, size in bytes, and direction (host to device or device to host).
    - Direction can be specified using `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`.

- `__restrict__`: A keyword that tells the compiler that the pointer is the only reference to that memory location, allowing for better optimization.
- `std::chrono`: C++ library for measuring time intervals, useful for benchmarking code performance.
- `std::vector`: C++ Standard Library container that encapsulates dynamic size arrays, providing automatic memory management and ease of use.
- `cudaEvent_t`: CUDA type used for timing GPU operations, allowing for precise measurement of kernel execution time.
- `cudaEventCreate()`: Function to create a CUDA event.
- `cudaEventRecord()`: Function to record an event at a specific point in the GPUs execution timeline.
- `cudaEventSynchronize()`: Function to wait for an event to complete.
- `cudaEventElapsedTime()`: Function to compute the elapsed time between two events, useful for measuring kernel execution time.


## Benchmarking
We can benchmark both CPU and GPU implementations of vector addition using `std::chrono` for CPU and `cudaEvent_t` for GPU. This allows us to compare the performance of both implementations.

```
Benchmarks from this code example:
GPU Time: 2.88832ms
Effective Bandwidth: 139.407 GB/s
CPU Time: 44.3468ms
Results match!
```

GPU time tells us how long the kernel took to execute on the GPU. Effective bandwidth is calculated based on the amount of data transferred and the time taken, giving us an idea of how efficiently the GPU is utilizing its memory bandwidth.

Our results show that the GPU implementation is significantly faster than the CPU implementation for vector addition, demonstrating the power of parallel processing on GPUs.

