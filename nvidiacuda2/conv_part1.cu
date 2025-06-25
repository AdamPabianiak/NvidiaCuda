#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) {          \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,            \
          cudaGetErrorString(e)); exit(EXIT_FAILURE);} } while(0)

// ----------------- CPU reference -----------------------------------------
void conv2dCPU(float* dst,
               const float* img, int W, int H,
               const float* ker, int K)
{
    int R = K / 2;            // radius
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int ky = -R; ky <= R; ++ky)
                for (int kx = -R; kx <= R; ++kx)
                {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < W && iy >= 0 && iy < H)
                        sum += img[iy * W + ix] *
                               ker[(ky + R) * K + (kx + R)];
                    // else zero-pad
                }
            dst[y * W + x] = sum;
        }
}

// ----------------- Naïve CUDA kernel -------------------------------------
__global__ void conv2dKernel(const float* img, float* dst,
                             const float* ker, int W, int H, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int R = K / 2;
    if (x >= W || y >= H) return;

    float sum = 0.f;
    for (int ky = -R; ky <= R; ++ky)
        for (int kx = -R; kx <= R; ++kx)
        {
            int ix = x + kx;
            int iy = y + ky;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H)
                sum += img[iy * W + ix] *
                       ker[(ky + R) * K + (kx + R)];
        }
    dst[y * W + x] = sum;
}

// ----------------- Helpers -----------------------------------------------
float maxError(const float* a, const float* b, size_t n)
{
    float m = 0.f;
    for (size_t i = 0; i < n; ++i)
        m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

void fillRandom(float* p, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        p[i] = static_cast<float>(rand()) / RAND_MAX;
}

// ----------------- Main ---------------------------------------------------
int main(int argc, char** argv)
{
    int W = (argc > 1) ? atoi(argv[1]) : 2048;
    int H = (argc > 2) ? atoi(argv[2]) : 2048;
    int K = (argc > 3) ? atoi(argv[3]) : 5;      // must be odd
    if (K % 2 == 0) { printf("Kernel must be odd\n"); return 0; }

    printf("Image %dx%d  |  Kernel %dx%d\n", W, H, K, K);

    size_t imgBytes = (size_t)W * H * sizeof(float);
    size_t kerBytes = (size_t)K * K * sizeof(float);

    // host buffers
    std::vector<float> h_img(W * H);
    std::vector<float> h_ker(K * K);
    std::vector<float> h_cpu(W * H);
    std::vector<float> h_gpu(W * H);

    fillRandom(h_img.data(), W * H);
    fillRandom(h_ker.data(), K * K);

    // --------------- CPU reference ----------------
    auto t0 = std::chrono::high_resolution_clock::now();
    conv2dCPU(h_cpu.data(), h_img.data(), W, H,
              h_ker.data(), K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double msCPU = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --------------- GPU buffers ------------------
    float *d_img, *d_dst, *d_ker;
    CUDA_CHECK(cudaMalloc(&d_img, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_dst, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_ker, kerBytes));

    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ker, h_ker.data(), kerBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    cudaEvent_t ev1, ev2;
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2));

    CUDA_CHECK(cudaEventRecord(ev1));
    conv2dKernel<<<grid, block>>>(d_img, d_dst, d_ker, W, H, K);
    CUDA_CHECK(cudaEventRecord(ev2));
    CUDA_CHECK(cudaEventSynchronize(ev2));
    float msGPU;
    CUDA_CHECK(cudaEventElapsedTime(&msGPU, ev1, ev2));

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_dst, imgBytes, cudaMemcpyDeviceToHost));

    // --------------- Verify -----------------------
    float err = maxError(h_cpu.data(), h_gpu.data(), W * H);
    bool ok = err < 1e-4f;

    printf("\nCPU  : %8.3f ms\n", msCPU);
    printf("GPU  : %8.3f ms\n", msGPU);
    printf("Speed: %.2fx\n", msCPU / msGPU);
    printf("Correct? %s (max |Δ| = %g)\n",
           ok ? "YES" : "NO", err);

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_ker));
    return 0;
}