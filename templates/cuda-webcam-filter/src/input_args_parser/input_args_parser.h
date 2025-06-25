#pragma once

#include <string>
#include <vector>
#include <cxxopts.hpp>

namespace cuda_filter
{

    enum class InputSource { WEBCAM, IMAGE, VIDEO, SYNTHETIC };
    enum class SyntheticPattern { CHECKERBOARD, GRADIENT, NOISE };

    struct FilterOptions
    {
        // Input
        InputSource inputSource;
        std::string inputPath;
        SyntheticPattern syntheticPattern;
        int deviceId;

        // Single filter legacy
        std::string filterType;
        int kernelSize;
        float sigma;
        float intensity;

        bool preview;

        // HDR controls
        float exposure;
        float gamma;
        float saturation;
        std::string tonemapper;

        // Pipeline & transitions
        std::vector<std::string> pipelineFilters;
        std::string              transitionType;
        float                    transitionProgress;
    };

    class InputArgsParser
    {
    public:
        InputArgsParser(int argc, char **argv);
        FilterOptions parseArgs();
    private:
        int m_argc; char **m_argv;
        void setupOptions(cxxopts::Options &options);
        InputSource      stringToInputSource(const std::string &str);
        SyntheticPattern stringToSyntheticPattern(const std::string &str);
    };

} // namespace cuda_filter
