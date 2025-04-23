#pragma once

#include <string>
#include <cxxopts.hpp>

namespace cuda_filter
{

    struct FilterOptions
    {
        int deviceId;
        std::string filterType;
        int kernelSize;
        float sigma;
        float intensity;
        bool preview;
    };

    class InputArgsParser
    {
    public:
        InputArgsParser(int argc, char **argv);

        FilterOptions parseArgs();

    private:
        int m_argc;
        char **m_argv;

        void setupOptions(cxxopts::Options &options);
    };

} // namespace cuda_filter
