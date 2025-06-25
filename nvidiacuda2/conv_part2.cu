#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) {                    \
  fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                  \
          cudaGetErrorString(e)); exit(EXIT_FAILURE); } } while (0)

// ---------------- CONSTANT MEMORY (max 7×7 = 49 coeffs) ------------------
__constant__ float d_KER[49];

// ---------------- SHARED-MEMORY TILED 2-D KERNEL --------------------------
template<int TILE>
__global__ void conv2dShared(const float* __restrict__ img,
                             float*       __restrict__ out,
                             int W, int H, int K)
{
    extern __shared__ float tile[];                 // (TILE+2R)² elements
    const int R   = K >> 1;
    const int shW = TILE + 2 * R;                   // shared-tile width

    int x  = blockIdx.x * TILE + threadIdx.x;       // global output coords
    int y  = blockIdx.y * TILE + threadIdx.y;

    int sx = threadIdx.x + R;                       // coords in shared tile
    int sy = threadIdx.y + R;

    // --- main load ---
    if (x < W && y < H)
        tile[sy * shW + sx] = img[y * W + x];
    else
        tile[sy * shW + sx] = 0.f;

    // --- halo loads (zero-pad) ---
    if (threadIdx.y < R) {
        int iy = y - R;
        int ty = sy - R;
        tile[ty * shW + sx] =
            (iy >= 0 && x < W) ? img[iy * W + x] : 0.f;
    }
    if (threadIdx.y >= TILE - R) {
        int iy = y + R;
        int ty = sy + R;
        tile[ty * shW + sx] =
            (iy < H && x < W) ? img[iy * W + x] : 0.f;
    }
    if (threadIdx.x < R) {
        int ix = x - R;
        int tx = sx - R;
        tile[sy * shW + tx] =
            (ix >= 0 && y < H) ? img[y * W + ix] : 0.f;
    }
    if (threadIdx.x >= TILE - R) {
        int ix = x + R;
        int tx = sx + R;
        tile[sy * shW + tx] =
            (ix < W && y < H) ? img[y * W + ix] : 0.f;
    }
    __syncthreads();

    if (x >= W || y >= H) return;

    float sum = 0.f;
    #pragma unroll
    for (int ky = 0; ky < K; ++ky)
        #pragma unroll
        for (int kx = 0; kx < K; ++kx)
            sum += tile[(sy + ky - R) * shW + (sx + kx - R)] *
                   d_KER[ky * K + kx];

    out[y * W + x] = sum;
}

// ---------------- SEPARABLE 1-D KERNEL (row or col) -----------------------
template<int TILE, bool HORIZ>
__global__ void conv1dSeparable(const float* __restrict__ img,
                                float*       __restrict__ out,
                                int W, int H, int K)
{
    extern __shared__ float tile[];
    const int R   = K >> 1;
    const int shL = TILE + 2 * R;

    int tid  = threadIdx.x;
    int base = blockIdx.x * TILE;

    if constexpr (HORIZ) {
        int y = blockIdx.y;                              // fixed row
        // load strip into shared
        for (int i = tid; i < shL; i += blockDim.x) {
            int g = base + i - R;
            tile[i] = (g >= 0 && g < W) ? img[y * W + g] : 0.f;
        }
        __syncthreads();
        int gx = base + tid;
        if (gx >= W) return;
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < K; ++k)
            acc += tile[tid + k] * d_KER[k];
        out[y * W + gx] = acc;
    } else {
        int x = blockIdx.y;                              // fixed column
        for (int i = tid; i < shL; i += blockDim.x) {
            int g = base + i - R;
            tile[i] = (g >= 0 && g < H) ? img[g * W + x] : 0.f;
        }
        __syncthreads();
        int gy = base + tid;
        if (gy >= H) return;
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < K; ++k)
            acc += tile[tid + k] * d_KER[k];
        out[gy * W + x] = acc;
    }
}

// ---------------- CPU reference (from Part 1) -----------------------------
void conv2dCPU(float* dst, const float* img, int W, int H,
               const float* ker, int K)
{
    int R = K >> 1;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float s = 0.f;
            for (int ky = -R; ky <= R; ++ky)
                for (int kx = -R; kx <= R; ++kx) {
                    int ix = x + kx, iy = y + ky;
                    if (ix >= 0 && ix < W && iy >= 0 && iy < H)
                        s += img[iy * W + ix] *
                             ker[(ky + R) * K + (kx + R)];
                }
            dst[y * W + x] = s;
        }
}

// ---------------- utility helpers ----------------------------------------
float maxErr(const float* a, const float* b, size_t n)
{
    float m = 0.f;
    for (size_t i = 0; i < n; ++i)
        m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}
void fillRand(float* p, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        p[i] = static_cast<float>(rand()) / RAND_MAX;
}

// launch helper for one TILE
template<int TILE>
float runShared(const float* d_in, float* d_out,
                int W, int H, int K, size_t shBytes)
{
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    conv2dShared<TILE><<<grid, block, shBytes>>>(
        d_in, d_out, W, H, K);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    return ms;
}

// -------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int W    = (argc > 1) ? atoi(argv[1]) : 4096;
    int H    = (argc > 2) ? atoi(argv[2]) : 4096;
    int K    = (argc > 3) ? atoi(argv[3]) : 5;    // must be odd ≤ 7
    int TILE = (argc > 4) ? atoi(argv[4]) : 16;   // 8 / 16 / 32 / 0=sweep

    if (K % 2 == 0 || K > 7) {
        printf("K must be odd <= 7 (constant mem limit).\n");
        return 0;
    }

    if (TILE != 0 && TILE != 8 && TILE != 16 && TILE != 32) {
        printf("TILE must be 8, 16, 32, or 0 to sweep.\n");
        return 0;
    }

    printf("Image %dx%d | Kernel %dx%d | TILE %d\n", W, H, K, K, TILE);

    size_t bytes = static_cast<size_t>(W) * H * sizeof(float);
    std::vector<float> h_in(W * H), h_outCPU(W * H), h_outGPU(W * H);
    std::vector<float> h_ker(K * K);

    fillRand(h_in.data(), W * H);
    fillRand(h_ker.data(), K * K);

    // --- CPU baseline ----------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    conv2dCPU(h_outCPU.data(), h_in.data(), W, H, h_ker.data(), K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double msCPU =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU buffers -----------------------------------------------------
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_KER, h_ker.data(), K * K * sizeof(float)));

    // --- Sweep mode ------------------------------------------------------
    if (TILE == 0) {
        for (int t : {8, 16, 32}) {
            size_t sh = (t + K - 1) * (t + K - 1) * sizeof(float);
            float ms = (t == 8)  ? runShared<8 >(d_in, d_out, W, H, K, sh) :
                        (t == 16) ? runShared<16>(d_in, d_out, W, H, K, sh) :
                                    runShared<32>(d_in, d_out, W, H, K, sh);
            printf("Tile %2d | GPU %.3f ms\n", t, ms);
        }
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        return 0;
    }

    // --- Single run ------------------------------------------------------
    size_t sh = (TILE + K - 1) * (TILE + K - 1) * sizeof(float);
    float msGPU = 0.f;
    if (TILE == 8)      msGPU = runShared<8 >(d_in, d_out, W, H, K, sh);
    else if (TILE == 16) msGPU = runShared<16>(d_in, d_out, W, H, K, sh);
    else                 msGPU = runShared<32>(d_in, d_out, W, H, K, sh);

    CUDA_CHECK(cudaMemcpy(h_outGPU.data(), d_out, bytes,
                          cudaMemcpyDeviceToHost));
    float err = maxErr(h_outCPU.data(), h_outGPU.data(), W * H);

    printf("\nCPU  : %.3f ms\nGPU  : %.3f ms\nSpeed : %.1fx\n"
           "Correct? %s (%.2g)\n",
           msCPU, msGPU, msCPU / msGPU,
           (err < 1e-4f) ? "YES" : "NO", err);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}