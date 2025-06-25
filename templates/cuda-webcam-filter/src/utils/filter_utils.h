#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace cuda_filter
{

    enum class FilterType
    {
        BLUR,
        SHARPEN,
        EDGE_DETECTION,
        EMBOSS,
        IDENTITY,
        HDR_TONEMAPPING   // ← added
    };

    class FilterUtils
    {
    public:
        static FilterType stringToFilterType(const std::string &filterName);
        static cv::Mat   createFilterKernel(FilterType type, int kernelSize, float intensity = 1.0f);

        static void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);

        // CPU fallback for HDR
        static void applyHDRToneMappingCPU(const cv::Mat &input,
                                           cv::Mat &output,
                                           float exposure,
                                           float gamma,
                                           float saturation,
                                           const std::string &algorithm);
    };

} // namespace cuda_filter
