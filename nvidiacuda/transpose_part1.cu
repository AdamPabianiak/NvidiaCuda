#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                                     \
    do {                                                                    \
        cudaError_t e = (err);                                              \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr,"CUDA ERROR %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(e));             \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------- CPU reference ------------------------------------------
void transposeCPU(float* dst, const float* src, int rows, int cols)
{
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst[c * rows + r] = src[r * cols + c];
}

// ---------------- Naïve GPU kernel ---------------------------------------
template<int BLOCK>
__global__ void transposeKernelNaive(float* dst, const float* src,
                                     int rows, int cols)
{
    int c = blockIdx.x * BLOCK + threadIdx.x;   // original column
    int r = blockIdx.y * BLOCK + threadIdx.y;   // original row

    if (r < rows && c < cols)
        dst[c * rows + r] = src[r * cols + c];
}

// ---------------- Utility ------------------------------------------------
void fillRandom(float* data, int n)
{
    for (int i = 0; i < n; ++i) data[i] = static_cast<float>(rand()) / RAND_MAX;
}

float maxError(const float* a, const float* b, int n)
{
    float m = 0.f;
    for (int i = 0; i < n; ++i)
        m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

// ---------------- Main ---------------------------------------------------
int main(int argc, char** argv)
{
    int rows = (argc > 1) ? atoi(argv[1]) : 2048;
    int cols = (argc > 2) ? atoi(argv[2]) : 1024;
    int  BS  = (argc > 3) ? atoi(argv[3]) : 16;          // block size (square)

    size_t bytesA = rows * cols * sizeof(float);
    size_t bytesB = cols * rows * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);      // CPU result
    float *h_Bgpu = (float*)malloc(bytesB);   // GPU result

    fillRandom(h_A, rows * cols);

    //----------------------------------------------------------------------
    // 1. CPU transpose & timing
    //----------------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    transposeCPU(h_B, h_A, rows, cols);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    //----------------------------------------------------------------------
    // 2. GPU transpose
    //----------------------------------------------------------------------
    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));

    dim3 block(BS, BS);
    dim3 grid((cols + BS - 1) / BS, (rows + BS - 1) / BS);

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaEventRecord(evStart));
    transposeKernelNaive<16><<<grid, block>>>(d_B, d_A, rows, cols);
    CUDA_CHECK(cudaEventRecord(evStop));

    CUDA_CHECK(cudaEventSynchronize(evStop));
    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, evStart, evStop));

    CUDA_CHECK(cudaMemcpy(h_Bgpu, d_B, bytesB, cudaMemcpyDeviceToHost));

    //----------------------------------------------------------------------
    // 3. Verify correctness
    //----------------------------------------------------------------------
    float err = maxError(h_B, h_Bgpu, rows * cols);
    bool ok = (err < 1e-5f);

    //----------------------------------------------------------------------
    // 4. Print results
    //----------------------------------------------------------------------
    printf("\nMATRIX %d x %d   block %dx%d\n", rows, cols, BS, BS);
    printf("CPU  transpose: %8.3f ms (%.2f GB/s)\n",
           cpu_ms,
           (bytesA + bytesB) / (cpu_ms * 1e6));
    printf("GPU  transpose: %8.3f ms (%.2f GB/s)\n",
           gpu_ms,
           (bytesA + bytesB) / (gpu_ms * 1e6));
    printf("Speed-up       : %.2fx\n", cpu_ms / gpu_ms);
    printf("Correct?       : %s (max |Δ| = %g)\n\n",
           ok ? "YES ✅" : "NO ❌", err);

    //----------------------------------------------------------------------
    // Cleanup
    //----------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A); free(h_B); free(h_Bgpu);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}