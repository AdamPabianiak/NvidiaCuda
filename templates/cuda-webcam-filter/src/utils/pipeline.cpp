#include "pipeline.h"
#include "kernels.h"
#include <plog/Log.h>
#include <cuda_runtime.h>

namespace cuda_filter {

FilterStage::FilterStage(FilterType t, const FilterOptions &o)
  : _type(t), _opts(o)
{
    if (_type != FilterType::HDR_TONEMAPPING)
        _kernel = FilterUtils::createFilterKernel(_type,_opts.kernelSize,_opts.intensity);
}

void FilterStage::apply(const cv::Mat &in, cv::Mat &out, cudaStream_t s) {
    if (_type==FilterType::HDR_TONEMAPPING)
        applyHDRToneMappingGPU(in,out,_opts.exposure,_opts.gamma,_opts.saturation,_opts.tonemapper);
    else
        applyFilterGPU(in,out,_kernel);
}

FilterPipeline::FilterPipeline() {}
FilterPipeline::~FilterPipeline(){
    for(auto &st:_streams) if(st) cudaStreamDestroy(st);
}

void FilterPipeline::addStage(FilterPtr st){ _stages.push_back(st); }
void FilterPipeline::clear(){
    _stages.clear(); _buffers.clear();
    for(auto&s:_streams) if(s) cudaStreamDestroy(s);
    _streams.clear();
}

void FilterPipeline::execute(const cv::Mat &in, cv::Mat &out){
    size_t n=_stages.size();
    if(!n){ in.copyTo(out); return; }
    _buffers.resize(n+1); _streams.resize(n);
    for(size_t i=0;i<n;++i)
        if(!_streams[i]) cudaStreamCreate(&_streams[i]);
    _buffers[0]=in;
    for(size_t i=0;i<n;++i){
        _buffers[i+1].create(in.size(),in.type());
        _stages[i]->apply(_buffers[i],_buffers[i+1],_streams[i]);
    }
    for(auto&s:_streams) cudaStreamSynchronize(s);
    _buffers[n].copyTo(out);
}

} // namespace
