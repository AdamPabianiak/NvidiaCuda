#include "input_args_parser.h"
#include <iostream>
#include "../utils/version.h"

namespace cuda_filter
{

    InputArgsParser::InputArgsParser(int argc, char **argv)
        : m_argc(argc), m_argv(argv)
    {
    }

    FilterOptions InputArgsParser::parseArgs()
    {
        cxxopts::Options options("cuda-webcam-filter", "Real-time webcam filter with CUDA acceleration");

        setupOptions(options);

        auto result = options.parse(m_argc, m_argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (result.count("version"))
        {
            std::cout << "CUDA Webcam Filter version " << CUDA_WEBCAM_FILTER_VERSION << std::endl;
            exit(0);
        }

        FilterOptions filterOptions;
        filterOptions.deviceId = result["device"].as<int>();
        filterOptions.filterType = result["filter"].as<std::string>();
        filterOptions.kernelSize = result["kernel-size"].as<int>();
        filterOptions.sigma = result["sigma"].as<float>();
        filterOptions.intensity = result["intensity"].as<float>();
        filterOptions.preview = result.count("preview") > 0;

        return filterOptions;
    }

    void InputArgsParser::setupOptions(cxxopts::Options &options)
    {
        options.add_options()("d,device", "Camera device ID", cxxopts::value<int>()->default_value("0"))("f,filter", "Filter type: blur, sharpen, edge, emboss", cxxopts::value<std::string>()->default_value("blur"))("k,kernel-size", "Kernel size for filters", cxxopts::value<int>()->default_value("3"))("s,sigma", "Sigma value for Gaussian blur", cxxopts::value<float>()->default_value("1.0"))("i,intensity", "Filter intensity", cxxopts::value<float>()->default_value("1.0"))("p,preview", "Show original video alongside filtered")("h,help", "Print usage")("v,version", "Print version information");
    }

} // namespace cuda_filter
