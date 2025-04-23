#include "webcam_handler.h"
#include <plog/Log.h>

namespace cuda_filter
{

    WebcamHandler::WebcamHandler(int deviceId) : m_cap(deviceId)
    {
        if (!m_cap.isOpened())
        {
            PLOG_ERROR << "Could not open camera device " << deviceId;
        }
        else
        {
            PLOG_INFO << "Camera initialized successfully";
        }
    }

    WebcamHandler::~WebcamHandler()
    {
        if (m_cap.isOpened())
        {
            m_cap.release();
        }
        cv::destroyAllWindows();
    }

    bool WebcamHandler::isOpened() const
    {
        return m_cap.isOpened();
    }

    bool WebcamHandler::readFrame(cv::Mat &frame)
    {
        return m_cap.read(frame);
    }

    void WebcamHandler::displayFrame(const cv::Mat &frame, const std::string &windowName)
    {
        cv::imshow(windowName, frame);
    }

    void WebcamHandler::displaySideBySide(const cv::Mat &original, const cv::Mat &filtered)
    {
        cv::Mat combined;
        cv::hconcat(original, filtered, combined);
        cv::imshow("Original | Filtered", combined);
    }

} // namespace cuda_filter
