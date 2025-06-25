#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>
#include <vector>
#include <algorithm>

namespace cuda_filter
{

#define CHECK_CUDA_ERROR(call)                                                    \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess) {                                                  \
            PLOG_ERROR << "CUDA error in " #call ": "                              \
                       << cudaGetErrorString(err);                                 \
            return;                                                                \
        }                                                                          \
    }

__global__ void convolutionKernel(const uchar3* __restrict__ in,
                                  uchar3* __restrict__ out,
                                  const float* __restrict__ ker,
                                  int w, int h, int c, int kS)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=w||y>=h) return;
    int r=kS/2, idx=(y*w+x)*c;
    for (int ch=0; ch<c; ++ch) {
        float sum=0.0f;
        for(int yy=-r;yy<=r;++yy)for(int xx=-r;xx<=r;++xx){
            int ix=clamp(x+xx,0,w-1), iy=clamp(y+yy,0,h-1);
            int iidx=(iy*w+ix)*c+ch;
            sum += ((float*)in)[iidx] * ker[(yy+r)*kS + (xx+r)];
        }
        unsigned char v = (unsigned char)clamp(sum,0.0f,255.0f);
        uchar3 &dst = ((uchar3*)out)[y*w+x];
        if (ch==0) dst.x = v;
        if (ch==1) dst.y = v;
        if (ch==2) dst.z = v;
    }
}

void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernelMat)
{
    if (input.empty()||kernelMat.empty()) {
        PLOG_ERROR << "Empty input/kernel";
        return;
    }
    output.create(input.size(), input.type());
    int w=input.cols, h=input.rows, c=input.channels(), kS=kernelMat.rows;
    size_t np=size_t(w)*h;

    static bool init=false;
    static uchar3 *d_in=nullptr,*d_out=nullptr;
    static float  *d_ker=nullptr;
    static int pw=0,ph=0,pc=0,pk=0;

    if (!init || pw!=w||ph!=h||pc!=c||pk!=kS) {
        if (init) {
            cudaFree(d_in); cudaFree(d_out); cudaFree(d_ker);
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_in,  np*sizeof(uchar3)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_out, np*sizeof(uchar3)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_ker, kS*kS*sizeof(float)));

        std::vector<float> hker(kS*kS);
        for(int i=0;i<kS;i++)for(int j=0;j<kS;j++)
            hker[i*kS+j]=kernelMat.at<float>(i,j);
        CHECK_CUDA_ERROR(cudaMemcpy(d_ker,hker.data(),
                             kS*kS*sizeof(float),
                             cudaMemcpyHostToDevice));

        init=true; pw=w;ph=h;pc=c;pk=kS;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_in,input.ptr<uchar3>(),
                        np*sizeof(uchar3),
                        cudaMemcpyHostToDevice));

    dim3 b(16,16), g((w+15)/16,(h+15)/16);
    convolutionKernel<<<g,b>>>(d_in,d_out,d_ker,w,h,c,kS);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(output.ptr<uchar3>(),d_out,
                        np*sizeof(uchar3),
                        cudaMemcpyDeviceToHost));
}

void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
{
    output.create(input.size(), input.type());
    cv::filter2D(input, output, -1, kernel);
}

} // namespace cuda_filter
