#include "input_args_parser.h"
#include <iostream>
#include <sstream>
#include "../utils/version.h"

namespace cuda_filter
{

InputArgsParser::InputArgsParser(int argc, char **argv)
    : m_argc(argc), m_argv(argv) {}

FilterOptions InputArgsParser::parseArgs()
{
    cxxopts::Options options("cuda-webcam-filter","Real-time webcam filter");
    setupOptions(options);
    auto result = options.parse(m_argc, m_argv);

    if (result.count("help"))    { std::cout<<options.help(); exit(0);}
    if (result.count("version")) { std::cout<<"Version "<<CUDA_WEBCAM_FILTER_VERSION<<"\n"; exit(0);}

    FilterOptions o;
    o.inputSource = stringToInputSource(result["input"].as<std::string>());
    o.inputPath   = result["path"].as<std::string>();
    if (o.inputSource==InputSource::SYNTHETIC)
        o.syntheticPattern = stringToSyntheticPattern(result["synthetic"].as<std::string>());
    else if (o.inputSource==InputSource::WEBCAM)
        o.deviceId = result["device"].as<int>();

    // legacy single-filter
    o.filterType = result["filter"].as<std::string>();
    o.kernelSize = result["kernel-size"].as<int>();
    o.sigma      = result["sigma"].as<float>();
    o.intensity  = result["intensity"].as<float>();
    o.preview    = result.count("preview")>0;

    // HDR
    o.exposure   = result["exposure"].as<float>();
    o.gamma      = result["gamma"].as<float>();
    o.saturation = result["saturation"].as<float>();
    o.tonemapper = result["tonemapper"].as<std::string>();

    // pipeline
    {
        std::stringstream ss(result["pipeline"].as<std::string>());
        std::string name;
        while (std::getline(ss,name,',')) if (!name.empty()) o.pipelineFilters.push_back(name);
    }
    o.transitionType     = result["transition"].as<std::string>();
    o.transitionProgress = result["transition-progress"].as<float>();

    return o;
}

void InputArgsParser::setupOptions(cxxopts::Options &options)
{
    options.add_options()
        ("i,input",        "webcam,image,video,synthetic",
            cxxopts::value<std::string>()->default_value("webcam"))
        ("p,path",         "Path to file",
            cxxopts::value<std::string>()->default_value("test_image.jpg"))
        ("s,synthetic",    "checkerboard,gradient,noise",
            cxxopts::value<std::string>()->default_value("checkerboard"))
        ("d,device",       "Camera ID", 
            cxxopts::value<int>()->default_value("0"))
        ("f,filter",       "blur,sharpen,edge,emboss,hdr",
            cxxopts::value<std::string>()->default_value("blur"))
        ("k,kernel-size",  "Kernel size",
            cxxopts::value<int>()->default_value("3"))
        ("sigma",          "Gaussian sigma",
            cxxopts::value<float>()->default_value("1.0"))
        ("intensity",      "Filter intensity",
            cxxopts::value<float>()->default_value("1.0"))
        ("preview",        "Show side-by-side")
        // HDR
        ("exposure",       "Exposure", 
            cxxopts::value<float>()->default_value("1.0"))
        ("gamma",          "Gamma",    
            cxxopts::value<float>()->default_value("2.2"))
        ("saturation",     "Saturation",
            cxxopts::value<float>()->default_value("1.0"))
        ("tonemapper",     "global/local",
            cxxopts::value<std::string>()->default_value("global"))
        // pipeline
        ("pipeline",       "Comma-separated filters",
            cxxopts::value<std::string>()->default_value("blur"))
        ("transition",     "none/wipe",
            cxxopts::value<std::string>()->default_value("none"))
        ("transition-progress","[0..1]",
            cxxopts::value<float>()->default_value("0.0"))
        ("h,help","Print help")
        ("v,version","Print version");
}

InputSource InputArgsParser::stringToInputSource(const std::string &str)
{ /* ... as before ... */ }
SyntheticPattern InputArgsParser::stringToSyntheticPattern(const std::string &str)
{ /* ... as before ... */ }

} // namespace cuda_filter
