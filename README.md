# CUDA Programming Exercises

Code, notes, and benchmark observations produced while studying *Programming Massively Parallel Processors: A Hands-on Approach*. The repository follows the book's progression from one-dimensional data parallelism to multidimensional thread mapping and baseline matrix multiplication.

This is a learning repository: early kernels intentionally favor clarity and correctness before introducing later optimization techniques.

## Progress

| Section | Implementation | Concepts |
| --- | --- | --- |
| Chapter 2 | [`chapter2/va.cu`](chapter2/va.cu) | CUDA execution model, host/device memory, grid-stride indexing, timing |
| Vector addition notes | [`vector_addition/`](vector_addition/) | Data parallelism, effective bandwidth, CPU/GPU comparison |
| Chapter 3 | [`chapter3/basic_mm.cu`](chapter3/basic_mm.cu) | 2D grids and blocks, row-major indexing, one-thread-per-output matrix multiplication |

## Requirements

- CUDA-capable NVIDIA GPU
- NVIDIA CUDA Toolkit with `nvcc`
- C++14-compatible host compiler

## Build and run

The repository currently keeps each exercise self-contained rather than using a root build system.

### Chapter 2: vector addition

```bash
nvcc -std=c++14 -O2 chapter2/va.cu -o chapter2/vector_add
./chapter2/vector_add
```

### Alternate vector-addition exercise

```bash
nvcc -std=c++14 -O2 vector_addition/va.cu -o vector_addition/vector_add
./vector_addition/vector_add
```

### Chapter 3: baseline matrix multiplication

```bash
nvcc -std=c++14 -O2 chapter3/basic_mm.cu -o chapter3/basic_mm
./chapter3/basic_mm
```

Executable syntax may differ on Windows. Run the generated `.exe` from PowerShell or Command Prompt when using the native Windows CUDA toolchain.

## What the exercises demonstrate

### One-dimensional data parallelism

The vector-addition kernels map one CUDA thread to one element and use CUDA events to isolate kernel execution time. The accompanying notes discuss grids, blocks, global thread indices, host/device transfers, and why simple vector operations are usually memory-bandwidth-bound.

### Two-dimensional thread mapping

The matrix-multiplication exercise maps `(threadIdx.x, threadIdx.y)` and `(blockIdx.x, blockIdx.y)` to one output matrix element:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

It is intentionally a baseline global-memory implementation with no shared-memory tiling. That makes it a useful correctness reference for later optimized kernels.

## Notes and measured results

Each chapter directory contains its own detailed notes:

- [`chapter2/README.md`](chapter2/README.md)
- [`vector_addition/README.md`](vector_addition/README.md)
- [`chapter3/README.md`](chapter3/README.md)

Reported timings are observations from the hardware used at the time, not portable performance guarantees. Re-run each executable on the target GPU before drawing comparisons.
