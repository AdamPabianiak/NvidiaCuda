#pragma once

#include <opencv2/opencv.hpp>

namespace cuda_filter
{

    class WebcamHandler
    {
    public:
        WebcamHandler(int deviceId);
        ~WebcamHandler();

        bool isOpened() const;
        bool readFrame(cv::Mat &frame);
        void displayFrame(const cv::Mat &frame, const std::string &windowName = "Filtered");
        void displaySideBySide(const cv::Mat &original, const cv::Mat &filtered);

    private:
        cv::VideoCapture m_cap;
    };

} // namespace cuda_filter
