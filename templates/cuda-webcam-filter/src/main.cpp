#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
#include "input_args_parser/input_args_parser.h"
#include "utils/webcam_handler.h"
#include "utils/filter_utils.h"
#include "kernels/kernels.h"

int main(int argc, char **argv)
{
    // Initialize logger
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    // Parse command line arguments
    cuda_filter::InputArgsParser parser(argc, argv);
    cuda_filter::FilterOptions options = parser.parseArgs();

    // Initialize webcam
    cuda_filter::WebcamHandler webcam(options.deviceId);
    if (!webcam.isOpened())
    {
        PLOG_ERROR << "Failed to open webcam";
        return -1;
    }

    // Create filter kernel
    cuda_filter::FilterType filterType = cuda_filter::FilterUtils::stringToFilterType(options.filterType);
    cv::Mat kernel = cuda_filter::FilterUtils::createFilterKernel(
        filterType, options.kernelSize, options.intensity);

    PLOG_INFO << "Filter: " << options.filterType
              << ", Kernel size: " << options.kernelSize
              << ", Intensity: " << options.intensity;

    cv::Mat frame, filtered;

    PLOG_INFO << "Press 'ESC' to exit";

    while (true)
    {
        // Capture frame
        if (!webcam.readFrame(frame))
        {
            PLOG_ERROR << "Failed to read frame";
            break;
        }

        // Apply filter using GPU
        cuda_filter::applyFilterGPU(frame, filtered, kernel);

        // Display results
        if (options.preview)
        {
            webcam.displaySideBySide(frame, filtered);
        }
        else
        {
            webcam.displayFrame(filtered);
        }

        // Exit on ESC key
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    PLOG_INFO << "Application terminated";
    return 0;
}
