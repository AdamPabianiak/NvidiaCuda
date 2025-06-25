#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)  do { cudaError_t e=(x); if(e){            \
  fprintf(stderr,"CUDA  %s:%d: %s\n",__FILE__,__LINE__,          \
          cudaGetErrorString(e)); exit(EXIT_FAILURE);} } while(0)

// ───────────────────────────────── CPU reference (unchanged) ───────────────
void transposeCPU(float* dst, const float* src, int R, int C)
{
  for(int r=0;r<R;++r) for(int c=0;c<C;++c) dst[c*R+r]=src[r*C+c];
}

// ───────────────────────────────── Naïve global-mem kernel ─────────────────
template<int TILE>
__global__ void transposeNaive(float* dst,const float* src,int R,int C)
{
  int c = blockIdx.x * TILE + threadIdx.x;
  int r = blockIdx.y * TILE + threadIdx.y;
  if(r<R && c<C) dst[c*R + r] = src[r*C + c];
}

// ───────────────────────────────── Tiled shared-mem kernel ─────────────────
template<int TILE>
__global__ void transposeTiledShared(float* dst,const float* src,int R,int C)
{
  __shared__ float tile[TILE][TILE+1];       // +1 avoids bank conflicts

  int c0 = blockIdx.x * TILE;
  int r0 = blockIdx.y * TILE;

  int c  = c0 + threadIdx.x;
  int r  = r0 + threadIdx.y;

  // ── Load from global to shared (coalesced) ──
  if(r<R && c<C) tile[threadIdx.y][threadIdx.x] = src[r*C + c];
  __syncthreads();

  // ── Write transposed part back (coalesced) ──
  c = r0 + threadIdx.x;     // swap indices
  r = c0 + threadIdx.y;

  if(r<C && c<R) dst[r*R + c] = tile[threadIdx.x][threadIdx.y];
}

// ───────────────────────────────── Helpers ────────────────────────────────
float wall_ms(std::chrono::high_resolution_clock::time_point a,
              std::chrono::high_resolution_clock::time_point b)
{
  return std::chrono::duration<float,std::milli>(b-a).count();
}

void fillRand(float* p,size_t n){ for(size_t i=0;i<n;++i) p[i]=(float)rand()/RAND_MAX; }

float maxError(float* a,float* b,size_t n)
{ float m=0; for(size_t i=0;i<n;++i) m=fmaxf(m,fabsf(a[i]-b[i])); return m; }

// ───────────────────────────────── Main ───────────────────────────────────
int main(int argc,char** argv)
{
  int R  = (argc>1)?atoi(argv[1]):2048;
  int C  = (argc>2)?atoi(argv[2]):1024;
  int TS = (argc>3)?atoi(argv[3]):32;             // TILE / block size (square)

  printf("Matrix %d×%d  |  Tile %d×%d  (Unified memory)\n",R,C,TS,TS);

  size_t bytesA = (size_t)R*C*sizeof(float);
  size_t bytesB = (size_t)C*R*sizeof(float);

  // ── Unified memory buffers ──
  float *A,*Bcpu,*Bgpu;
  CUDA_CHECK(cudaMallocManaged(&A   , bytesA));
  CUDA_CHECK(cudaMallocManaged(&Bgpu, bytesB));
  Bcpu = (float*)malloc(bytesB);

  fillRand(A,R*C);
  cudaDeviceSynchronize();   // ensure data ready

  // ───────────────── CPU reference ─────────────────
  auto t0 = std::chrono::high_resolution_clock::now();
  transposeCPU(Bcpu,A,R,C);
  auto t1 = std::chrono::high_resolution_clock::now();
  float msCPU = wall_ms(t0,t1);

  // ───────────────── GPU kernels & timing ──────────
  dim3 block(TS,TS);
  dim3 grid((C+TS-1)/TS,(R+TS-1)/TS);

  cudaEvent_t ev1,ev2; cudaEventCreate(&ev1); cudaEventCreate(&ev2);

  // Naïve
  cudaEventRecord(ev1);
  transposeNaive<32><<<grid,block>>>(Bgpu,A,R,C);   // template arg 32 fine—threads beyond TILE unused
  cudaEventRecord(ev2); cudaEventSynchronize(ev2);
  float msNaive; cudaEventElapsedTime(&msNaive,ev1,ev2);

  // Optimized (shared-memory)
  cudaEventRecord(ev1);
  switch(TS){
    case 8 : transposeTiledShared<8 ><<<grid,block>>>(Bgpu,A,R,C); break;
    case 16: transposeTiledShared<16><<<grid,block>>>(Bgpu,A,R,C); break;
    case 32: transposeTiledShared<32><<<grid,block>>>(Bgpu,A,R,C); break;
    default: printf("Tile %d not specialised - choose 8/16/32\n",TS); return 0;
  }
  cudaEventRecord(ev2); cudaEventSynchronize(ev2);
  float msOpt;  cudaEventElapsedTime(&msOpt,ev1,ev2);

  cudaDeviceSynchronize();   // ensure Bgpu filled

  // ───────────────── Correctness ──────────────────
  float err = maxError(Bcpu,Bgpu,(size_t)R*C);
  printf("Max |Δ| vs CPU: %.3g  →  %s\n",err,(err<1e-5)?"OK ✅":"BAD ❌");

  double gB = (bytesA+bytesB)/1e9;
  printf("\n%-22s %8.3f ms  |  %.2f GB/s\n","CPU",  msCPU , gB/(msCPU*1e-3));
  printf("%-22s %8.3f ms  |  %.2f GB/s\n","Naïve GPU", msNaive, gB/(msNaive*1e-3));
  printf("%-22s %8.3f ms  |  %.2f GB/s\n","Optimized GPU", msOpt , gB/(msOpt*1e-3));

  printf("\nSpeed-up (GPU naïve  / CPU)      : %.1fx\n", msCPU / msNaive);
  printf("Speed-up (GPU optimized / naïve)  : %.1fx\n", msNaive / msOpt);

  // ── Clean up ──
  CUDA_CHECK(cudaFree(A)); CUDA_CHECK(cudaFree(Bgpu)); free(Bcpu);
  return 0;
}