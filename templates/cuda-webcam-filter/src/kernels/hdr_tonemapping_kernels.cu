#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "kernels.h"
#include "filter_utils.h"

namespace cuda_filter
{

__global__ void hdrToneMapKernel(const uchar3* __restrict__ in,
                                 uchar3* __restrict__ out,
                                 int cols, int rows,
                                 float exposure,
                                 float invGamma,
                                 float saturation)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=cols||y>=rows) return;
    int idx = y*cols + x;
    uchar3 p = in[idx];
    float b=p.x/255.0f, g=p.y/255.0f, r=p.z/255.0f;
    float L = 0.0722f*b + 0.7152f*g + 0.2126f*r;
    float Le = L*exposure;
    float Ld = Le/(1.0f+Le);
    float ratio = (L>1e-6f)?(Ld/L):0.0f;
    float rr=r*ratio, gg=g*ratio, bb=b*ratio;
    rr=powf(rr, invGamma); gg=powf(gg, invGamma); bb=powf(bb, invGamma);
    float gray=r*0.299f+g*0.587f+b*0.114f;
    rr = gray + saturation*(rr-gray);
    gg = gray + saturation*(gg-gray);
    bb = gray + saturation*(bb-gray);
    out[idx].x = (uchar)(fminf(fmaxf(bb,0.0f),1.0f)*255.0f);
    out[idx].y = (uchar)(fminf(fmaxf(gg,0.0f),1.0f)*255.0f);
    out[idx].z = (uchar)(fminf(fmaxf(rr,0.0f),1.0f)*255.0f);
}

void applyHDRToneMappingGPU(const cv::Mat &input,
                            cv::Mat &output,
                            float exposure,
                            float gamma,
                            float saturation,
                            const std::string &algorithm)
{
    static bool initialized = false;
    static cudaStream_t stream;
    static uchar3 *d_in=nullptr, *d_out=nullptr;
    static int prevC=0, prevR=0;

    int cols = input.cols, rows = input.rows;
    size_t np = size_t(cols)*rows;

    if (!initialized || cols!=prevC||rows!=prevR) {
        if (initialized) {
            cudaFree(d_in); cudaFree(d_out);
            cudaStreamDestroy(stream);
        }
        cudaStreamCreate(&stream);
        cudaMalloc(&d_in,  np*sizeof(uchar3));
        cudaMalloc(&d_out, np*sizeof(uchar3));
        initialized=true; prevC=cols; prevR=rows;
    }
    output.create(input.size(), input.type());
    cudaMemcpyAsync(d_in, input.ptr<uchar3>(), np*sizeof(uchar3),
                    cudaMemcpyHostToDevice, stream);
    dim3 b(16,16), g((cols+15)/16,(rows+15)/16);
    float invG=1.0f/gamma;
    hdrToneMapKernel<<<g,b,0,stream>>>(d_in,d_out,cols,rows,exposure,invG,saturation);
    cudaMemcpyAsync(output.ptr<uchar3>(), d_out, np*sizeof(uchar3),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

} // namespace cuda_filter
