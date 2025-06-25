#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace cuda_filter
{

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);

    // New HDR GPU entrypoint
    void applyHDRToneMappingGPU(const cv::Mat &input,
                                cv::Mat &output,
                                float exposure,
                                float gamma,
                                float saturation,
                                const std::string &algorithm);

    namespace cuda {
#ifdef __CUDACC__
        __host__ __device__ inline int divUp(int a, int b) {
            return (a + b - 1) / b;
        }
#endif
    }

} // namespace cuda_filter
