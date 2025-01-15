
# Chapter 3 — Matrix Multiplication

*(Programming Massively Parallel Processors: A Hands-on Approach)*

> These notes are written while studying Chapter 3 and implementing the **basic matrix multiplication kernel**, without any optimizations that are introduced in later chapters. The goal here is correctness, mapping, and understanding multidimensional data—not peak performance tuning yet.

---

## Why Matrix Multiplication Appears in Chapter 3

Up to Chapter 2, everything was **1D**:

* vectors,
* linear indexing,
* `blockIdx.x`, `threadIdx.x`.

Matrix multiplication is the first *real* problem where **1D thinking breaks down naturally**.

Matrices are **2D data**:

* rows
* columns

Chapter 3 introduces **multidimensional grids and multidimensional indexing**, and matrix multiplication is the perfect example to force that mental shift .

---

## Matrix Multiplication Refresher (Sequential View)

Given:

* `A` of size `M × K`
* `B` of size `K × N`
* `C` of size `M × N`

Each element:

```
C[row][col] = sum over k ( A[row][k] * B[k][col] )
```

Key observation (very important for CUDA thinking):

> **Each output element `C[row][col]` is independent.**

That immediately suggests **data parallelism**:

* one thread → one output element.

This matches exactly how I structured my kernel.

---

## Output-Centric Decomposition (as Used in the Book)

Chapter 3 frames matrix multiplication using **output-centric decomposition**:

* Each thread is responsible for **one element of C**
* Threads compute:

  * their `(row, col)`
  * loop over `k`
  * accumulate a scalar result

No synchronization needed because:

* threads never write to the same output element .

This is exactly the strategy used in my kernel.

---

## Mapping Threads to 2D Matrix Coordinates

This is the *core learning objective* of Chapter 3.

### 2D Grid + 2D Block

Instead of thinking in 1D:

```
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

We now think in **2D**:

```
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

This mapping:

* mirrors matrix layout
* keeps code readable
* avoids manual flattening mistakes

This exact mapping is used in my kernel.

---

## My Kernel: Direct Mapping to Chapter 3 Concepts

From `basic_mm.cu`:

```cpp
__global__ void matMulKernel(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

Everything here is **pure Chapter 3**:

* multidimensional grid
* multidimensional indexing
* linearized memory access
* boundary checks
* no shared memory
* no tiling
* no architectural assumptions

This is the **baseline matrix multiplication** described in Section 3.4 of the book .

---

## Why Linear Indexing Still Exists

Even though matrices are conceptually 2D, CUDA global memory is **linear**.

So we manually convert:

* `A[row][k]` → `A[row * K + k]`
* `B[k][col]` → `B[k * N + col]`
* `C[row][col]` → `C[row * N + col]`

This matches the book’s explanation that **multidimensional data is mapped onto linear memory**, while multidimensional grids help with indexing clarity .

---

## Kernel Launch Configuration (Host Code)

Again, this is straight from Chapter 3:

```cpp
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(
    (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (M + threadsPerBlock.y - 1) / threadsPerBlock.y
);

matMulKernel<<<numBlocks, threadsPerBlock>>>(A, B, C, M, K, N);
```

Key points:

* Grid dimensions correspond to **output matrix dimensions**
* Each block covers a **tile-shaped region of C**
* Boundary checks handle non-multiples cleanly

No mention yet of why 16×16 is good or bad—that discussion belongs to later chapters.

---

## Performance Observations (From My Runs)

From running the executable:

```
Average time over 10 iterations: ~17 ms
Performance: ~560–580 GFLOPS
Matrix multiplication successful!
```

Important takeaway **at this stage of the book**:

> The goal here is **correct mapping and correctness**, not peak performance.

The book explicitly introduces this version as a **baseline**, meant to be improved later once:

* memory behavior,
* caching,
* execution hardware

are properly understood (all future chapters).

---

## Why This Version Is Intentionally “Naive”

This kernel:

* loads from global memory repeatedly
* recomputes redundant values
* has no data reuse

And that is **intentional**.

Chapter 3 is teaching:

* *how to think in 2D*
* *how threads map to data*
* *how grids and blocks align with problem geometry*

Optimization comes later. This is the **reference implementation** everything else will be compared against .

