# CUDA C++ Programming: Hands-on Lab Guide

This lab guide will walk you through a series of exercises designed to help you understand the fundamentals of CUDA C++ programming. By modifying and experimenting with the provided code examples, you'll gain hands-on experience with GPU programming concepts.

- [CUDA C++ Programming: Hands-on Lab Guide](#cuda-c-programming-hands-on-lab-guide)
  - [Exercise 1: Hello World from the GPU](#exercise-1-hello-world-from-the-gpu)
    - [Tasks:](#tasks)
  - [Exercise 2: Understanding Thread and Block Indices](#exercise-2-understanding-thread-and-block-indices)
    - [Tasks:](#tasks-1)
  - [Exercise 3: Single Block Loop Parallelization](#exercise-3-single-block-loop-parallelization)
    - [Tasks:](#tasks-2)
  - [Exercise 4: Multi-Block Loop Parallelization](#exercise-4-multi-block-loop-parallelization)
    - [Tasks:](#tasks-3)
  - [Exercise 5: Error Handling in CUDA](#exercise-5-error-handling-in-cuda)
    - [Tasks:](#tasks-4)
  - [Exercise 6: Optimizing with Grid-Stride Loops](#exercise-6-optimizing-with-grid-stride-loops)
    - [Tasks:](#tasks-5)
    - [Questions:](#questions)
  - [Exercise 7: Memory Prefetching and Initialization](#exercise-7-memory-prefetching-and-initialization)
    - [Tasks:](#tasks-6)
    - [Questions:](#questions-1)
  - [Exercise 8: Thread Configuration and Performance Analysis](#exercise-8-thread-configuration-and-performance-analysis)
    - [Tasks:](#tasks-7)
    - [Questions:](#questions-2)
  - [Exercise 9: CPU vs GPU Performance Comparison](#exercise-9-cpu-vs-gpu-performance-comparison)
    - [Tasks:](#tasks-8)
    - [Questions:](#questions-3)
  - [Exercise 10: Memory Optimization with Prefetching](#exercise-10-memory-optimization-with-prefetching)
    - [Tasks:](#tasks-9)
    - [Questions:](#questions-4)
  - [Exercise 11: GPU Initialization and Error Handling](#exercise-11-gpu-initialization-and-error-handling)
    - [Tasks:](#tasks-10)
    - [Questions:](#questions-5)
  - [Exercise 12: Matrix Multiplication Optimization](#exercise-12-matrix-multiplication-optimization)
    - [Tasks:](#tasks-11)
    - [Questions:](#questions-6)
  - [Exercise 13: CPU vs GPU Initialization](#exercise-13-cpu-vs-gpu-initialization)
    - [Tasks:](#tasks-12)
    - [Questions:](#questions-7)
  - [Exercise 14: Understanding CUDA Streams Basics](#exercise-14-understanding-cuda-streams-basics)
    - [Tasks:](#tasks-13)
    - [Questions:](#questions-8)
  - [Exercise 15: Stream-Based Initialization](#exercise-15-stream-based-initialization)
    - [Tasks:](#tasks-14)
    - [Questions:](#questions-9)
  - [Exercise 16: Stream-Sliced Vector Addition](#exercise-16-stream-sliced-vector-addition)
    - [Tasks:](#tasks-15)
    - [Questions:](#questions-10)
  - [Exercise 17: Advanced Stream Synchronization](#exercise-17-advanced-stream-synchronization)
    - [Tasks:](#tasks-16)
    - [Questions:](#questions-11)
  - [Exercise 18: Multi-Stage Pipeline with Streams](#exercise-18-multi-stage-pipeline-with-streams)
    - [Tasks:](#tasks-17)
    - [Questions:](#questions-12)
  - [Exercise 19: Optimizing Block Count Based on SM Count](#exercise-19-optimizing-block-count-based-on-sm-count)
    - [Tasks:](#tasks-18)
    - [Questions:](#questions-13)
  - [Exercise 20: Understanding Unified Memory Behavior](#exercise-20-understanding-unified-memory-behavior)
    - [Tasks:](#tasks-19)
    - [Questions:](#questions-14)
  - [Exercise 21: Memory Prefetching Optimization](#exercise-21-memory-prefetching-optimization)
    - [Tasks:](#tasks-20)
    - [Questions:](#questions-15)

## Exercise 1: Hello World from the GPU

**File: [hello-world-gpu.cu](examples/1-gpu-hello-world/hello-world-gpu.cu)**

This first example demonstrates the basic structure of a CUDA program with both CPU and GPU functions.

```c
#include <stdio.h>

void helloCPU()
{
    printf("Hello from the CPU.\n");
}

/*
 * The addition of `__global__` signifies that this function
 * should be launced on the GPU.
 */

__global__ void helloGPU()
{
    printf("Hello from the GPU.\n");
}

int main()
{
    helloCPU();

    /*
     * Add an execution configuration with the <<<...>>> syntax
     * will launch this function as a kernel on the GPU.
     */

    helloGPU<<<1, 1>>>();

    /*
     * `cudaDeviceSynchronize` will block the CPU stream until
     * all GPU kernels have completed.
     */

    cudaDeviceSynchronize();
}
```

### Tasks:

1. Compile and run the program as is. Observe the output.
   ```
   nvcc hello-world-gpu.cu -o hello-gpu
   ./hello-gpu
   ```

2. Modify the program to print "Hello from the GPU" 5 times by changing the execution configuration to `<<<1, 5>>>`. 
   - What happens and why?

3. Change the execution configuration to `<<<5, 1>>>`.
   - How does the output differ from the previous configuration?
   - What does this tell you about the relationship between blocks and threads?

4. Experiment with removing `cudaDeviceSynchronize()`.
   - What happens when you run the program without it?
   - Why do you think this happens?

5. Create a new GPU function `helloGPU2` that prints "Hello again from the GPU". Call it after `helloGPU` with different execution configurations.
   - Try to make `helloGPU2` print before `helloGPU`

## Exercise 2: Understanding Thread and Block Indices

**File: [thread-and-block-idx.cu](examples/2-cuda-kernel-idx/thread-and-block-idx.cu)**

This example shows how to access thread and block indices within a kernel.

```c
#include <stdio.h>

__global__ void printHelloForCorrectExecutionConfiguration()
{
    if (threadIdx.x == 1023 && blockIdx.x == 255)
    {
        printf("Hello from GPU thread!\n");
    }
}

int main()
{
    // TODO: Change kernel parameters to print Hello message
    printHelloForCorrectExecutionConfiguration<<<2, 4>>>();

    cudaDeviceSynchronize();
}
```

### Tasks:

1. The current code doesn't print anything. Examine the if-condition in the kernel. What execution configuration would make this kernel print "Hello from GPU thread!"?

2. Modify the execution configuration to make the kernel print the message.

3. Change the kernel to print the thread ID and block ID for all threads:
   ```c
   printf("Block: %d, Thread: %d\n", blockIdx.x, threadIdx.x);
   ```
   - Run with `<<<2, 4>>>` and observe the output
   - What do you notice about the order of execution?

4. Modify the kernel to only print when `threadIdx.x + blockIdx.x` equals 3.
   - Run with `<<<4, 4>>>` and count how many times the message prints
   - Explain why you get that number of messages

5. Add conditions to make only the first thread in each block print a message.
   - How would you identify the first thread in a block?
   - How would you identify the last thread in a block if the block size is variable?

## Exercise 3: Single Block Loop Parallelization

**File: [single-block-loop-gpu.cu](examples/3-loops/single-block-loop-gpu.cu)**

This example demonstrates how to parallelize a loop using a single block of threads.

```c
#include <stdio.h>

__global__ void loop()
{
    /*
     * This kernel does the work of only 1 iteration
     * of the original for loop. Indication of which
     * "iteration" is being executed by this kernel is
     * still available via `threadIdx.x`.
     */

    printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
    loop<<<1, 10>>>();
    cudaDeviceSynchronize();
}
```

### Tasks:

1. Compile and run the program. Notice how each thread executes what would be a single iteration of a loop.

2. Modify the execution configuration to use 20 threads instead of 10.
   - What is the maximum number of threads you can have in a single block on your GPU?
   - Find this limit using `cudaGetDeviceProperties`

3. Add code to make the kernel perform some computation:
   ```c
   int result = threadIdx.x * threadIdx.x;
   printf("Thread %d computed: %d\n", threadIdx.x, result);
   ```

4. Create a new version that processes an array:
   - Define a global array `int results[1000]`
   - In the kernel, have each thread write its square to its position in the array
   - Add code to print the array after kernel execution
   - What's the problem with this approach? (Hint: think about scope and memory)

5. Fix the previous task by using proper CUDA memory management:
   ```c
   int *results;
   cudaMallocManaged(&results, 1000 * sizeof(int));
   // Initialize array values to 0
   // Launch kernel
   // Print values
   // Free memory
   ```

## Exercise 4: Multi-Block Loop Parallelization

**File: [multiple-block-loop-gpu.cu](examples/3-loops/multiple-block-loop-gpu.cu)**

This example shows how to use multiple blocks of threads to parallelize a loop.

```c
#include <stdio.h>

__global__ void loop()
{
    /*
     * This idiomatic expression gives each thread
     * a unique index within the entire grid.
     */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", i);
}

int main()
{
    loop<<<5, 5>>>();
    cudaDeviceSynchronize();
}
```

### Tasks:

1. Compile and run the program. How many numbers are printed? Verify that they go from 0 to 24.

2. Change the execution configuration to `<<<10, 10>>>`. 
   - How many numbers will this print?
   - What's the pattern of values?

3. Modify the kernel to print only even numbers:
   ```c
   if (i % 2 == 0) {
       printf("%d\n", i);
   }
   ```

4. Implement a grid-stride loop that allows a fixed number of threads to process a larger array:
   ```c
   __global__ void gridStrideLoop(int *data, int n)
   {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       int stride = blockDim.x * gridDim.x;
       
       for (int i = idx; i < n; i += stride) {
           data[i] = i * i; // Square each element
       }
   }
   ```
   - Launch this with a small grid (e.g., `<<<2, 8>>>`) to process a large array (e.g., 100 elements)
   - Print the results to verify correctness

5. Compare the performance of a single large grid versus a grid-stride approach:
   - For an array of 1,000,000 elements, try:
     - Single launch with enough threads to cover all elements
     - Grid-stride with a smaller number of threads (e.g., 256 threads total)
   - Use `cudaEvent` timing to measure kernel execution time
   - Which is faster and why?


## Exercise 5: Error Handling in CUDA

**File:** [double-elements-gpu.cu](examples/4-allocation/double-elements-gpu.cu)

CUDA applications need proper error handling to ensure robustness. In this exercise, you'll modify a simple CUDA program to include comprehensive error checking.

### Tasks:

1. **Create an error checking helper function:**
   - Add a helper function that checks CUDA error codes and prints meaningful messages:
   ```c
   inline cudaError_t checkCuda(cudaError_t result) {
       if (result != cudaSuccess) {
           fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
           // Optionally add: assert(result == cudaSuccess);
       }
       return result;
   }
   ```

2. **Add error checking for memory allocation:**
   - Modify the `cudaMallocManaged` call to use your helper function:
   ```c
   checkCuda(cudaMallocManaged(&a, size));
   ```

3. **Add error checking for kernel launch:**
   - Add code after the kernel launch to check for synchronous errors:
   ```c
   cudaError_t syncErr = cudaGetLastError();
   if (syncErr != cudaSuccess) {
       printf("Kernel launch error: %s\n", cudaGetErrorString(syncErr));
   }
   ```

4. **Add error checking for kernel execution:**
   - Add code to capture asynchronous errors during kernel execution:
   ```c
   cudaError_t asyncErr = cudaDeviceSynchronize();
   if (asyncErr != cudaSuccess) {
       printf("Kernel execution error: %s\n", cudaGetErrorString(asyncErr));
   }
   ```

5. **Test with deliberate errors:**
   - Try launching the kernel with an invalid execution configuration (e.g., too many threads per block) and observe the error messages.
   - Study [error-handling.cu](examples/6-errors/error-handling.cu) to see how errors are caught and reported.

## Exercise 6: Optimizing with Grid-Stride Loops

**File:** [double-elements-gpu.cu](examples/4-allocation/double-elements-gpu.cu)

Grid-stride loops allow CUDA threads to process multiple elements of an array, enabling efficient handling of large datasets with a limited number of threads.

### Tasks:

1. **Increase the array size:**
   - Modify the program to handle a much larger array (e.g., N = 10,000,000).
   - Run the program and observe if it still works correctly.

2. **Add timing code:**
   - Add CUDA event timing to measure kernel execution time:
   ```c
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   // Kernel launch here
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel execution time: %f ms\n", milliseconds);
   ```

3. **Implement a grid-stride loop:**
   - Study the implementation in [grid-stride-double.cu](examples/5-grid-stride/grid-stride-double.cu).
   - Modify the `doubleElements` kernel in your file to use a grid-stride loop:
   ```c
   __global__ void doubleElements(int *a, int N) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       int stride = blockDim.x * gridDim.x;
       
       for (int i = idx; i < N; i += stride) {
           a[i] *= 2;
       }
   }
   ```

4. **Experiment with different grid sizes:**
   - Test the performance with various block counts (e.g., 32, 64, 128, 256).
   - Keep the number of threads per block constant (e.g., 256).
   - Record the execution time for each configuration.

5. **Optimize the number of blocks:**
   - Add code to query the device for its number of streaming multiprocessors (SMs):
   ```c
   int deviceId;
   cudaGetDevice(&deviceId);
   int numberOfSMs;
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
   ```
   - Set the number of blocks to a multiple of the SM count (e.g., 2× or 4× the SM count).
   - Compare performance with your previous configurations.

### Questions:
- How does the execution time change with different numbers of blocks?
- Why is it beneficial to use a number of blocks that is a multiple of the SM count?
- What happens to performance if you use too few or too many blocks?

## Exercise 7: Memory Prefetching and Initialization

**File:** [block-config.cu](examples/4-allocation/block-config.cu)

CUDA Unified Memory can benefit significantly from prefetching data to the appropriate device before it's needed.

### Tasks:

1. **Add GPU initialization:**
   - Create a new kernel to initialize the array on the GPU:
   ```c
   __global__ void initializeElementsToGPU(int initialValue, int *a, int N) {
       int i = threadIdx.x + blockIdx.x * blockDim.x;
       int stride = blockDim.x * gridDim.x;
       
       for (int j = i; j < N; j += stride) {
           a[j] = initialValue;
       }
   }
   ```

2. **Add timing for both CPU and GPU initialization:**
   - Implement timing for both CPU and GPU initialization methods.
   - For CPU timing, use standard C `clock()`:
   ```c
   clock_t start = clock();
   // CPU initialization here
   clock_t end = clock();
   double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // in ms
   ```
   - For GPU timing, use CUDA events as shown in Exercise 2.

3. **Add memory prefetching:**
   - Add code to prefetch the array to the GPU before initialization:
   ```c
   int deviceId;
   cudaGetDevice(&deviceId);
   cudaMemPrefetchAsync(a, size, deviceId);
   ```
   - Add timing to measure the impact of prefetching.

4. **Prefetch back to CPU for verification:**
   - Add code to prefetch the array back to the CPU before verification:
   ```c
   cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);
   ```
   - Measure and compare verification time with and without this prefetch.

5. **Experiment with different array sizes:**
   - Test with various array sizes (e.g., 10K, 100K, 1M, 10M elements).
   - Record initialization and verification times for each size.
   - Determine at what array size GPU initialization becomes more efficient than CPU initialization.

### Questions:
- How does memory prefetching affect the performance of your program?
- At what array size does GPU initialization become faster than CPU initialization?
- What happens if you don't prefetch the data back to the CPU before verification?

## Exercise 8: Thread Configuration and Performance Analysis

**Files:** [double-elements-gpu.cu](examples/4-allocation/double-elements-gpu.cu) and [block-config.cu](examples/4-allocation/block-config.cu)

Understanding how thread configurations affect performance is crucial for optimizing CUDA applications.

### Tasks:

1. **Modify the programs to accept command-line arguments:**
   - Allow the user to specify array size, number of blocks, and threads per block:
   ```c
   int N = (argc > 1) ? atoi(argv[1]) : 1000;
   size_t threads_per_block = (argc > 2) ? atoi(argv[2]) : 256;
   size_t number_of_blocks = (argc > 3) ? atoi(argv[3]) : ((N + threads_per_block - 1) / threads_per_block);
   ```

2. **Create a performance testing loop:**
   - Write a script or modify your program to automatically test different configurations:
     - Various array sizes (e.g., 10K, 100K, 1M, 10M)
     - Different thread counts per block (e.g., 32, 64, 128, 256, 512, 1024)
     - Different block counts or calculation methods

3. **Add detailed timing measurements:**
   - Measure and report:
     - Memory allocation time
     - Data initialization time
     - Kernel execution time
     - Memory freeing time
     - Total application time

4. **Analyze occupancy:**
   - Add code to calculate theoretical occupancy based on your thread configuration:
   ```c
   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, deviceId);
   int maxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
   int maxBlocksPerMultiProcessor = props.maxBlocksPerMultiProcessor;
   int threadsPerSM = threads_per_block * number_of_blocks / props.multiProcessorCount;
   float occupancyPercentage = (float)threadsPerSM / maxThreadsPerMultiProcessor * 100.0f;
   printf("Theoretical occupancy: %.2f%%\n", occupancyPercentage);
   ```

### Questions:
- What is the optimal thread configuration for different array sizes?
- How does occupancy correlate with performance?
- What are the trade-offs between using many blocks with few threads vs. few blocks with many threads?

## Exercise 9: CPU vs GPU Performance Comparison

**Files:** [vector-add-cpu.cu](examples/7-vector/vector-add-cpu.cu) and [vector-add-gpu.cu](examples/7-vector/vector-add-gpu.cu)

These files implement vector addition on CPU and GPU respectively. This exercise will help you understand the performance differences between CPU and GPU implementations.

### Tasks:

1. **Run both implementations and record the execution times:**
   - Compile and run both programs:
   ```bash
   nvcc -o vector-add-cpu vector-add-cpu.cu
   nvcc -o vector-add-gpu vector-add-gpu.cu
   ./vector-add-cpu
   ./vector-add-gpu
   ```
   - Record the execution times for both implementations

2. **Modify both programs to try different array sizes:**
   - Change the `N` value in both programs to:
     - 2^20 (small)
     - 2^24 (medium)
     - 2^28 (large - already set)
   - For each size, record the execution times for both CPU and GPU implementations
   - Create a table or graph showing the relationship between array size and speedup (CPU time / GPU time)

3. **Analyze the performance crossover point:**
   - Find the approximate array size where GPU begins to outperform CPU
   - In your own words, explain why small arrays might not benefit from GPU acceleration

4. **Modify the GPU kernel to experiment with different block sizes:**
   - Try thread counts per block of: 32, 64, 128, 256, 512, 1024
   - Keep the total number of blocks formula the same: `(N + threadsPerBlock - 1) / threadsPerBlock`
   - Record the execution time for each configuration
   - Which block size gives the best performance? Why do you think that is?

### Questions:
- What factors contribute to the GPU outperforming the CPU for large arrays?
- What are the overheads associated with GPU computing that might make it slower for small arrays?
- How does the choice of block size affect performance?

## Exercise 10: Memory Optimization with Prefetching

**Files:** [vector-add-no-prefetch.cu](examples/09-vector-add-prefetch/vector-add-no-prefetch.cu) and [vector-add-prefetch.cu](examples/09-vector-add-prefetch/vector-add-prefetch.cu)

These files demonstrate the impact of memory prefetching on performance. This exercise will help you understand how memory movement affects GPU program performance.

### Tasks:

1. **Compare the execution time with and without prefetching:**
   - Run both implementations and record the execution times
   - Use the NVIDIA profiler to get more insight into memory operations:
   ```bash
   nsys profile --stats=true ./vector-add-no-prefetch
   nsys profile --stats=true ./vector-add-prefetch
   ```

2. **Modify the prefetching implementation:**
   - Add prefetching of result array back to the CPU before verification:
   ```c
   cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
   ```
   - Add this code before the `checkElementsAre` call
   - Measure execution time and compare with the previous version

3. **Create a hybrid prefetching strategy:**
   - Create a new version of the code that only prefetches part of each array
   - For example, prefetch the first half of each array to the GPU
   - Measure the impact on performance

4. **Add timing for memory operations:**
   - Add CUDA event timing to measure:
     - Initialization time
     - Prefetching time
     - Kernel execution time
     - Verification time
   - Use the following pattern:
   ```c
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   // Operation to time
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Operation time: %f ms\n", milliseconds);
   ```

### Questions:
- How does prefetching affect the overall execution time?
- Which operation benefits most from prefetching?
- In what scenarios might prefetching be less beneficial?
- How does the memory transfer time compare to computation time?

## Exercise 11: GPU Initialization and Error Handling

**Files:** [vector-add-no-prefetch.cu](examples/09-vector-add-prefetch/vector-add-no-prefetch.cu) and [vector-add-prefetch.cu](examples/09-vector-add-prefetch/vector-add-prefetch.cu)

In this exercise, you'll improve the vector addition programs by adding GPU-based initialization and robust error handling.

### Tasks:

1. **Create a GPU initialization kernel:**
   - Add a new kernel to initialize arrays on the GPU:
   ```c
   __global__ void initWithGPU(float num, float *a, int N)
   {
     int index = threadIdx.x + blockIdx.x * blockDim.x;
     int stride = blockDim.x * gridDim.x;
     
     for (int i = index; i < N; i += stride)
     {
       a[i] = num;
     }
   }
   ```
   - Replace CPU initialization with GPU initialization
   - Add timing to compare CPU vs GPU initialization speed

2. **Implement comprehensive error handling:**
   - Study the `checkCuda` function in `vector-add-gpu.cu`
   - Add error checking for all CUDA operations including:
     - Memory allocation
     - Memory prefetching (if used)
     - Kernel launches
     - Device synchronization
     - Memory freeing

3. **Test error detection:**
   - Intentionally create an error condition (e.g., specify too many threads per block)
   - Verify that your error handling catches and reports the error
   - Restore the correct configuration after testing

4. **Add device property queries:**
   - Query and print additional device properties:
   ```c
   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, deviceId);
   printf("Device: %s\n", props.name);
   printf("Compute capability: %d.%d\n", props.major, props.minor);
   printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
   printf("Max threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
   ```
   - Use these properties to optimize your block size

### Questions:
- How much faster is GPU initialization compared to CPU initialization?
- What are the most important device properties to consider when optimizing kernel launch parameters?
- What types of errors can occur in CUDA programs, and how can they be handled effectively?

## Exercise 12: Matrix Multiplication Optimization

**Files:** [matrix-multiply-2d-cpu.cu](examples/8-matrix-multiply/matrix-multiply-2d-cpu.cu) and [matrix-multiply-2d-gpu.cu](examples/8-matrix-multiply/matrix-multiply-2d-gpu.cu)

These files implement matrix multiplication on CPU and GPU. This exercise will have you optimize the GPU implementation for better performance.

### Tasks:

1. **Compare CPU and GPU performance:**
   - Run both implementations and record execution times
   - Try different matrix sizes (adjust the N definition):
     - N = 512
     - N = 1024 (default)
     - N = 2048 (be cautious with larger sizes)

2. **Add memory prefetching:**
   - Add code to prefetch matrices to the GPU before computation:
   ```c
   int deviceId;
   cudaGetDevice(&deviceId);
   cudaMemPrefetchAsync(a, size, deviceId);
   cudaMemPrefetchAsync(b, size, deviceId);
   cudaMemPrefetchAsync(c_gpu, size, deviceId);
   ```
   - Measure the impact on performance

3. **Implement a tiled matrix multiplication:**
   - Create a new kernel that uses shared memory for tiling:
   ```c
   __global__ void matrixMulTiled(int *a, int *b, int *c, int N)
   {
       __shared__ int aTile[TILE_SIZE][TILE_SIZE];
       __shared__ int bTile[TILE_SIZE][TILE_SIZE];
       
       int row = blockIdx.x * blockDim.x + threadIdx.x;
       int col = blockIdx.y * blockDim.y + threadIdx.y;
       
       int sum = 0;
       
       // Loop over tiles
       for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
           // Load tiles into shared memory
           if (row < N && t * TILE_SIZE + threadIdx.y < N)
               aTile[threadIdx.x][threadIdx.y] = a[row * N + t * TILE_SIZE + threadIdx.y];
           else
               aTile[threadIdx.x][threadIdx.y] = 0;
               
           if (t * TILE_SIZE + threadIdx.x < N && col < N)
               bTile[threadIdx.x][threadIdx.y] = b[(t * TILE_SIZE + threadIdx.x) * N + col];
           else
               bTile[threadIdx.x][threadIdx.y] = 0;
               
           __syncthreads();
           
           // Compute partial sum for this tile
           for (int i = 0; i < TILE_SIZE; i++)
               sum += aTile[threadIdx.x][i] * bTile[i][threadIdx.y];
               
           __syncthreads();
       }
       
       if (row < N && col < N)
           c[row * N + col] = sum;
   }
   ```
   - Define `TILE_SIZE` as 16 or 32
   - Replace the original kernel with this tiled version
   - Compare performance with the original implementation

4. **Experiment with different tile sizes:**
   - Try tile sizes of 8, 16, and 32
   - Record performance for each size
   - Find the optimal tile size for your GPU

### Questions:
- How does tiling affect the performance of matrix multiplication?
- Why does shared memory help with matrix multiplication performance?
- What factors determine the optimal tile size?
- How does the GPU speedup for matrix multiplication compare to the speedup for vector addition?

## Exercise 13: CPU vs GPU Initialization

**Files to use:** [init-kernel-cpu.cu](examples/10-init-kernel/init-kernel-cpu.cu) and [init-kernel-gpu.cu](examples/10-init-kernel/init-kernel-gpu.cu)

In this exercise, you'll compare initializing data on the CPU versus the GPU and understand the performance implications.

### Tasks:

1. **Compile and run both versions:**
   ```bash
   nvcc -o init-cpu init-kernel-cpu.cu
   nvcc -o init-gpu init-kernel-gpu.cu
   ./init-cpu
   ./init-gpu
   ```

2. **Add timing code to measure initialization performance:**
   - For both files, add CUDA event timing around the initialization sections:
   ```c
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   // CPU or GPU initialization here
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Initialization time: %f ms\n", milliseconds);
   ```

3. **Experiment with different array sizes:**
   - Modify both programs to test with these array sizes:
     - N = 2<<20 (small)
     - N = 2<<24 (medium - current setting)
     - N = 2<<28 (large)
   - Record initialization times for each size

4. **Analyze prefetching impact:**
   - In `init-kernel-cpu.cu`, move the prefetching code before the CPU initialization
   - Measure the performance difference and explain why it changed

### Questions:
- At what array size does GPU initialization become more efficient than CPU initialization?
- How does prefetching affect the performance of CPU initialization? Why?
- What happens if you remove the prefetching entirely? Test and explain.

## Exercise 14: Understanding CUDA Streams Basics

**Files to use:** [print-numbers-sync.cu](examples/11-stream-intro/print-numbers-sync.cu) and [print-numbers-async.cu](examples/11-stream-intro/print-numbers-async.cu)

These files demonstrate basic stream operations in CUDA. You'll modify them to understand how streams affect execution order.

### Tasks:

1. **Run both versions and observe the output ordering:**
   ```bash
   nvcc -o print-sync print-numbers-sync.cu
   nvcc -o print-async print-numbers-async.cu
   ./print-sync
   ./print-async
   ```
   - Note any differences in the output order

2. **Modify the async version:**
   - Instead of creating and destroying a stream for each number, create a fixed array of 3 streams:
   ```c
   cudaStream_t streams[3];
   for (int i = 0; i < 3; ++i) {
       cudaStreamCreate(&streams[i]);
   }
   
   for (int i = 0; i < 5; ++i) {
       printNumber<<<1, 1, 0, streams[i % 3]>>>(i);
   }
   
   cudaDeviceSynchronize();
   
   for (int i = 0; i < 3; ++i) {
       cudaStreamDestroy(streams[i]);
   }
   ```

3. **Add artificial delay to the kernel:**
   - Modify the `printNumber` kernel to include a delay:
   ```c
   __global__ void printNumber(int number) {
       // Add a small artificial delay
       int wait = 1000000;
       while(wait--) { }
       printf("%d\n", number);
   }
   ```
   - Run again and observe any changes in output order

4. **Create a priority stream version:**
   - Create streams with different priorities:
   ```c
   int priority_high, priority_low;
   cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
   
   cudaStream_t streamHigh, streamLow;
   cudaStreamCreateWithPriority(&streamHigh, cudaStreamNonBlocking, priority_high);
   cudaStreamCreateWithPriority(&streamLow, cudaStreamNonBlocking, priority_low);
   
   // Launch odd numbers on high priority
   // Launch even numbers on low priority
   for (int i = 0; i < 10; ++i) {
       if (i % 2 == 0)
           printNumber<<<1, 1, 0, streamLow>>>(i);
       else
           printNumber<<<1, 1, 0, streamHigh>>>(i);
   }
   ```

### Questions:
- How does the execution order differ between synchronous and asynchronous versions?
- Why might the output order be unpredictable even with separate streams?
- What impact does stream priority have? Is it guaranteed to execute in priority order?

## Exercise 15: Stream-Based Initialization

**Files to use:** [stream-init.cu](examples/12-stream-init/stream-init.cu) and [no-stream-init.cu](examples/12-stream-init/no-stream-init.cu)

This exercise focuses on the performance benefits of using streams for parallel initialization.

### Tasks:

1. **Add comprehensive timing code to both versions:**
   - Measure and record times for:
     - Memory allocation
     - Initialization of all three arrays
     - Prefetching
     - Vector addition
     - Verification
     - Overall execution

2. **Modify `no-stream-init.cu` to use streams:**
   - Replace the three sequential initialization calls with three streams like in `stream-init.cu`
   - Make sure to create/destroy streams properly

3. **Run nvprof to capture timeline information:**
   ```bash
   nsys profile --stats=true ./no-stream-init
   nsys profile --stats=true ./stream-init
   ```
   - Compare the timelines to see the difference in kernel execution patterns

4. **Experiment with stream count:**
   - Create a new version that uses 4 streams instead of 3
   - Use the 4th stream for the addition operation
   - Compare performance with the 3-stream version

### Questions:
- What performance improvement do you observe when using streams for initialization?
- Does using a separate stream for addition provide any benefit? Why or why not?
- What happens if you run the initialization kernels in the default stream but the addition kernel in a non-default stream?

## Exercise 16: Stream-Sliced Vector Addition

**File to use:** [stream-sliced.cu](examples/12-stream-init/stream-sliced.cu)

This exercise introduces a different approach to streaming: dividing the data into chunks and processing each chunk in its own stream.

### Tasks:

1. **Understand the existing code:**
   - Trace through the code and identify:
     - How the data is divided
     - How streams are assigned
     - The data processing flow

2. **Modify the stream count:**
   - Change the number of streams from 8 to:
     - 2 streams
     - 4 streams
     - 16 streams
   - Measure performance for each configuration

3. **Add error checking:**
   - Add proper error checking after each CUDA operation
   - Create a helper function for error checking:
   ```c
   inline void checkCuda(cudaError_t result) {
       if (result != cudaSuccess) {
           fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
           exit(EXIT_FAILURE);
       }
   }
   ```

4. **Implement overlapping computation and memory transfers:**
   - Modify the code to start copying results back to the host as soon as each chunk is processed
   - Use a loop structure like:
   ```c
   for (int i = 0; i < numberOfStreams; ++i) {
       // Initialize data
       initWith<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(3, a[i], streamN);
       initWith<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(4, b[i], streamN);
       initWith<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(0, c[i], streamN);
       
       // Process data
       addVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(c[i], a[i], b[i], streamN);
       
       // Start copying back to host (will be overlapped with next iteration's processing)
       cudaMemcpyAsync(h_c[i], c[i], streamSize, cudaMemcpyDeviceToHost, streams[i]);
   }
   ```

5. **Profile with different data sizes:**
   - Test with N = 2<<22, 2<<24, and 2<<26
   - Record performance metrics for each size

### Questions:
- What is the optimal number of streams for your GPU? Why?
- How does the performance of stream-sliced processing compare to processing the entire array at once?
- What factors limit the scalability of the stream-sliced approach?

## Exercise 17: Advanced Stream Synchronization 

For this exercise, create a new file named `stream-sync.cu` that demonstrates stream dependencies.

### Tasks:

1. **Create a program with the following structure:**
   - Allocate two input arrays and one output array
   - Create three streams: `streamA`, `streamB`, and `streamC`
   - Initialize array A in streamA and array B in streamB
   - Create CUDA events to mark completion of each initialization
   - Make streamC wait for both events before executing the vector addition kernel
   
   ```c
   cudaStream_t streamA, streamB, streamC;
   cudaStreamCreate(&streamA);
   cudaStreamCreate(&streamB);
   cudaStreamCreate(&streamC);
   
   cudaEvent_t eventA, eventB;
   cudaEventCreate(&eventA);
   cudaEventCreate(&eventB);
   
   // Initialize arrays in separate streams
   initWith<<<blocks, threads, 0, streamA>>>(3, a, N);
   initWith<<<blocks, threads, 0, streamB>>>(4, b, N);
   
   // Record events when initialization is complete
   cudaEventRecord(eventA, streamA);
   cudaEventRecord(eventB, streamB);
   
   // Make streamC wait for both initializations to complete
   cudaStreamWaitEvent(streamC, eventA, 0);
   cudaStreamWaitEvent(streamC, eventB, 0);
   
   // Launch addition in streamC
   addVectorsInto<<<blocks, threads, 0, streamC>>>(c, a, b, N);
   ```

2. **Add timing to measure:**
   - Initialization time for each array
   - Time between launching the init kernels and the addition kernel
   - Total execution time

3. **Create a comparison version without dependencies:**
   - Remove the event waiting
   - Launch the addition kernel in the default stream
   - Compare performance and correctness

4. **Implement priority-based scheduling:**
   - Modify the stream creation to use priorities
   - Assign higher priority to the addition kernel's stream
   - Measure the impact on performance

### Questions:
- How does the performance of the event-synchronized version compare to running kernels sequentially?
- What are the potential risks of not synchronizing streams when there are data dependencies?
- How much overhead do CUDA events add to the execution time?

## Exercise 18: Multi-Stage Pipeline with Streams

Create a new file `pipeline.cu` that implements a multi-stage data processing pipeline using CUDA streams.

### Tasks:

1. **Implement a three-stage pipeline:**
   - Stage 1: Initialize data (random numbers)
   - Stage 2: Process data (e.g., square each element)
   - Stage 3: Reduce data (e.g., sum all elements)

2. **Use stream synchronization to ensure correct execution order:**
   - Each stage must wait for the previous stage to complete
   - Different batches of data can be processed concurrently

3. **Process multiple batches of data:**
   - Divide a large array into smaller batches
   - Process each batch through the pipeline
   - Use multiple streams to overlap execution of different batches

4. **Compare with a sequential implementation:**
   - Implement the same functionality without streams
   - Measure and compare the performance

5. **Optimize for your GPU:**
   - Determine the optimal batch size and number of streams
   - Experiment with grid and block sizes
   - Profile and identify bottlenecks

### Questions:
- How much performance improvement does the pipelined approach provide?
- What is the optimal number of batches for your GPU?
- What factors limit the performance of the pipelined approach?

## Exercise 19: Optimizing Block Count Based on SM Count

**File to use:** [vector-add-SM-blocks.cu](examples/13-vector-add-sm-blocks/vector-add-SM-blocks.cu)

This exercise will help you understand how to optimize CUDA execution configuration based on the GPU's streaming multiprocessor (SM) count.

### Tasks:

1. **Fix the bug in the block count calculation:**
   - The current code has a bug in the `numberOfBlocks` calculation. It's trying to use `numberOfBlocks` to calculate itself!
   - Add code to query the number of SMs:
   ```c
   cudaGetDevice(&deviceId);
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
   printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);
   ```
   - Fix the calculation to make `numberOfBlocks` a multiple of the SM count:
   ```c
   numberOfBlocks = 32 * numberOfSMs; // Use 32 blocks per SM
   ```

2. **Add performance timing:**
   - Add CUDA event timing to measure kernel execution time:
   ```c
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   addArraysInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel execution time: %f ms\n", milliseconds);
   ```

3. **Experiment with different multipliers:**
   - Try different multipliers for the SM count: 1, 2, 4, 8, 16, 32, 64
   - Record the execution time for each configuration
   - Create a table or graph of your results

4. **Experiment with different thread counts:**
   - Try different thread counts per block: 128, 256, 512, 1024
   - For each thread count, use your best SM multiplier from the previous step
   - Record execution times and compare

### Questions:
- Why is it beneficial to use a number of blocks that's a multiple of the SM count?
- What happens to performance if you use too few blocks? Too many?
- Is there an optimal threads-per-block size for your GPU? Why?

## Exercise 20: Understanding Unified Memory Behavior

**File to use:** [page-faults.cu](examples/14-unified-memory-page-faults/page-faults.cu)

This exercise will help you understand how Unified Memory behaves under different access patterns and how page faulting affects performance.

### Tasks:

1. **Experiment with GPU-only access:**
   - Complete the code to run only the GPU kernel:
   ```c
   // Experiment 1: GPU-only access
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   deviceKernel<<<256, 256>>>(a, N);
   cudaDeviceSynchronize();
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("GPU-only access time: %f ms\n", milliseconds);
   ```
   - Profile this with `nsys profile --stats=true ./my_program`
   - Analyze the memory operations in the output

2. **Experiment with CPU-only access:**
   - Reset the code and run only the CPU function:
   ```c
   // Experiment 2: CPU-only access
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   hostFunction(a, N);
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("CPU-only access time: %f ms\n", milliseconds);
   ```
   - Profile this and analyze the memory operations

3. **Experiment with GPU-then-CPU access:**
   - Run the GPU kernel followed by the CPU function:
   ```c
   // Experiment 3: GPU then CPU access
   deviceKernel<<<256, 256>>>(a, N);
   cudaDeviceSynchronize();
   hostFunction(a, N);
   ```
   - Profile and analyze memory transfers

4. **Experiment with CPU-then-GPU access:**
   - Run the CPU function followed by the GPU kernel:
   ```c
   // Experiment 4: CPU then GPU access
   hostFunction(a, N);
   deviceKernel<<<256, 256>>>(a, N);
   cudaDeviceSynchronize();
   ```
   - Profile and analyze memory transfers

5. **Analyze and document your findings:**
   - For each experiment, record:
     - Number of page faults
     - Direction of memory transfers (HtoD or DtoH)
     - Total memory transferred
     - Execution time

### Questions:
- When do page faults occur in each scenario?
- How does the initial accessor of memory affect subsequent memory transfers?
- What pattern of memory access would be most efficient for a real application?
- How does the size of the data affect the page fault behavior?

## Exercise 21: Memory Prefetching Optimization

**File to use:** [vector-add-prefetch.cu](examples/15-unified-memory-prefetch/vector-add-prefetch.cu)

This exercise will show you how to use asynchronous memory prefetching to optimize performance by reducing page faults.

### Tasks:

1. **Add memory prefetching to GPU:**
   - Add code to prefetch arrays to the GPU before kernel execution:
   ```c
   // Prefetch arrays to the GPU
   cudaMemPrefetchAsync(a, size, deviceId);
   cudaMemPrefetchAsync(b, size, deviceId);
   cudaMemPrefetchAsync(c, size, deviceId);
   ```
   - Add this code after initialization but before kernel launch
   - Profile and measure the performance impact

2. **Add memory prefetching back to CPU:**
   - Add code to prefetch the result array back to the CPU before verification:
   ```c
   // Prefetch result back to CPU for verification
   cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
   ```
   - Add this after kernel execution but before verification
   - Profile and measure the impact

3. **Add comprehensive timing:**
   - Add timing for each phase of execution:
     - Memory allocation
     - Initialization
     - Prefetching to GPU
     - Kernel execution
     - Prefetching to CPU
     - Verification
   - Record and compare times for each phase

4. **Experiment with partial prefetching:**
   - Create a version that only prefetches part of each array
   - For example, prefetch the first half of each array:
   ```c
   cudaMemPrefetchAsync(a, size/2, deviceId);
   ```
   - Compare performance with full prefetching

5. **Try different array sizes:**
   - Modify the code to test with different array sizes:
     - N = 2<<20 (small)
     - N = 2<<24 (medium - current)
     - N = 2<<28 (large)
   - For each size, compare performance with and without prefetching

### Questions:
- At what data size does prefetching provide the most significant benefit?
- Which operation benefits most from prefetching?
- How does partial prefetching affect performance compared to full prefetching?
- How would you decide whether to use prefetching in a real application?
