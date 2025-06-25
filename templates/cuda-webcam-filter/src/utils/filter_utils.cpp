#include "filter_utils.h"
#include <plog/Log.h>
#include <algorithm>
#include <cmath>

namespace cuda_filter
{

    FilterType FilterUtils::stringToFilterType(const std::string &filterName)
    {
        if (filterName == "blur")      return FilterType::BLUR;
        if (filterName == "sharpen")   return FilterType::SHARPEN;
        if (filterName == "edge")      return FilterType::EDGE_DETECTION;
        if (filterName == "emboss")    return FilterType::EMBOSS;
        if (filterName == "identity")  return FilterType::IDENTITY;
        if (filterName == "hdr" ||
            filterName == "tonemap" ||
            filterName == "tone_mapping")
                                        return FilterType::HDR_TONEMAPPING;
        PLOG_WARNING << "Unknown filter type: " << filterName << ". Using blur.";
        return FilterType::BLUR;
    }

    cv::Mat FilterUtils::createFilterKernel(FilterType type, int kernelSize, float intensity)
    {
        if (type == FilterType::HDR_TONEMAPPING)
            return cv::Mat();

        if (kernelSize % 2 == 0) ++kernelSize;
        cv::Mat kernel;
        switch (type)
        {
            case FilterType::BLUR:
                kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) /
                         float(kernelSize * kernelSize);
                break;
            case FilterType::SHARPEN:
                kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
                kernel.at<float>(kernelSize/2, kernelSize/2) = 1.0f + 4.0f*intensity;
                if (kernelSize >= 3) {
                    kernel.at<float>(kernelSize/2-1, kernelSize/2) = -intensity;
                    kernel.at<float>(kernelSize/2+1, kernelSize/2) = -intensity;
                    kernel.at<float>(kernelSize/2, kernelSize/2-1) = -intensity;
                    kernel.at<float>(kernelSize/2, kernelSize/2+1) = -intensity;
                }
                break;
            case FilterType::EDGE_DETECTION:
                kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
                if (kernelSize >= 3) {
                    float v = -intensity;
                    for (int r=0; r<3; ++r)
                        for (int c=0; c<3; ++c)
                            kernel.at<float>(r,c) = (r==1 && c==1) ? 8.0f*intensity : v;
                }
                break;
            case FilterType::EMBOSS:
                kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
                if (kernelSize >= 3) {
                    kernel.at<float>(0,0) = -2.0f*intensity; kernel.at<float>(0,1) = -1.0f*intensity;
                    kernel.at<float>(1,0) = -1.0f*intensity; kernel.at<float>(1,1) =  1.0f;
                    kernel.at<float>(1,2) =  1.0f*intensity; kernel.at<float>(2,1) =  1.0f*intensity;
                    kernel.at<float>(2,2) =  2.0f*intensity;
                }
                break;
            case FilterType::IDENTITY:
            default:
                kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
                kernel.at<float>(kernelSize/2, kernelSize/2) = 1.0f;
                break;
        }
        return kernel;
    }

    void FilterUtils::applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        cv::filter2D(input, output, -1, kernel);
    }

    void FilterUtils::applyHDRToneMappingCPU(const cv::Mat &input,
                                             cv::Mat &output,
                                             float exposure,
                                             float gamma,
                                             float saturation,
                                             const std::string &algorithm)
    {
        // Convert to float
        cv::Mat inF;
        input.convertTo(inF, CV_32FC3, 1.0f/255.0f);
        output.create(input.size(), input.type());
        cv::Mat outF(input.size(), CV_32FC3);

        for (int y = 0; y < inF.rows; ++y) {
            const cv::Vec3f* inRow = inF.ptr<cv::Vec3f>(y);
            cv::Vec3f*       oRow = outF.ptr<cv::Vec3f>(y);
            for (int x = 0; x < inF.cols; ++x) {
                cv::Vec3f pix = inRow[x];
                float L = pix[0]*0.2126f + pix[1]*0.7152f + pix[2]*0.0722f;
                float Le = L*exposure;
                float Ld = Le/(1.0f+Le);
                float ratio = (L>1e-6f)?(Ld/L):0.0f;
                cv::Vec3f tone = pix*ratio;
                float invG = 1.0f/gamma;
                tone[0] = std::pow(tone[0], invG);
                tone[1] = std::pow(tone[1], invG);
                tone[2] = std::pow(tone[2], invG);
                float gray = pix[0]*0.299f + pix[1]*0.587f + pix[2]*0.114f;
                for (int c=0; c<3; ++c)
                    tone[c] = gray + saturation*(tone[c]-gray);
                for (int c=0; c<3; ++c)
                    tone[c] = std::clamp(tone[c], 0.0f, 1.0f);
                oRow[x] = tone;
            }
        }
        outF.convertTo(output, CV_8UC3, 255.0f);
    }

} // namespace cuda_filter
