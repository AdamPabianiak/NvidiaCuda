#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Log.h>
#include "input_args_parser/input_args_parser.h"
#include "utils/input_handler.h"
#include "utils/pipeline.h"
#include "filter_utils.h"

using namespace cuda_filter;

int main(int argc, char **argv)
{
    plog::ConsoleAppender<plog::TxtFormatter> app;
    plog::init(plog::info, &app);

    InputArgsParser parser(argc,argv);
    FilterOptions opts = parser.parseArgs();

    std::ostringstream ss;
    for (int i=0;i<opts.pipelineFilters.size();++i){
        if(i) ss<<","; ss<<opts.pipelineFilters[i];
    }
    PLOG_INFO << "Pipeline: "<<ss.str()
              <<" | Trans: "<<opts.transitionType
              <<" | Prog: "<<opts.transitionProgress;

    InputHandler ih(opts);
    if(!ih.isOpened()){ PLOG_ERROR<<"Cannot open input"; return -1; }

    // Build pipeline(s)
    FilterPipeline pipe, A, B;
    bool doWipe = opts.transitionType=="wipe" && opts.pipelineFilters.size()==2;
    if(doWipe){
        auto f1 = stringToFilterType(opts.pipelineFilters[0]);
        A.addStage(std::make_shared<FilterStage>(f1,opts));
        auto f2 = stringToFilterType(opts.pipelineFilters[1]);
        B.addStage(std::make_shared<FilterStage>(f2,opts));
    } else {
        for(auto &n:opts.pipelineFilters){
            auto ft = stringToFilterType(n);
            pipe.addStage(std::make_shared<FilterStage>(ft,opts));
        }
    }

    cv::Mat frame, o1, o2, out;
    PLOG_INFO<<"Press ESC to quit";

    while(true){
        if(!ih.readFrame(frame)) break;
        double t0 = cv::getTickCount();

        if(doWipe){
            A.execute(frame,o1);
            B.execute(frame,o2);
            int split=int(frame.cols*opts.transitionProgress);
            out = o1.clone();
            o2(cv::Rect(split,0,frame.cols-split,frame.rows))
              .copyTo(out(cv::Rect(split,0,frame.cols-split,frame.rows)));
        } else {
            pipe.execute(frame,out);
        }

        double t1=cv::getTickCount();
        double ms=(t1-t0)/cv::getTickFrequency()*1000.0;
        double fps=1000.0/ms;

        std::string info = "FPS: "+std::to_string(int(fps))+" ms: "+std::to_string(ms).substr(0,5);
        cv::putText(out,info,{10,30},cv::FONT_HERSHEY_SIMPLEX,0.7,{255,255,0},2);
        if(opts.preview) ih.displaySideBySide(frame,out);
        else ih.displayFrame(out);

        if(cv::waitKey(1)==27) break;
    }

    PLOG_INFO<<"Done";
    return 0;
}
