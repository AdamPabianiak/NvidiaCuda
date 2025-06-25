#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "filter_utils.h"
#include "input_args_parser.h"

namespace cuda_filter {

class IFilterStage {
public:
    virtual ~IFilterStage() = default;
    virtual void apply(const cv::Mat &in, cv::Mat &out, cudaStream_t) = 0;
};

using FilterPtr = std::shared_ptr<IFilterStage>;

class FilterStage : public IFilterStage {
public:
    FilterStage(FilterType type, const FilterOptions &opts);
    void apply(const cv::Mat &in, cv::Mat &out, cudaStream_t) override;
private:
    FilterType    _type;
    FilterOptions _opts;
    cv::Mat       _kernel;
};

class FilterPipeline {
public:
    FilterPipeline();
    ~FilterPipeline();
    void addStage(FilterPtr);
    void clear();
    void execute(const cv::Mat &in, cv::Mat &out);
private:
    std::vector<FilterPtr>    _stages;
    std::vector<cv::Mat>      _buffers;
    std::vector<cudaStream_t> _streams;
};

} // namespace
