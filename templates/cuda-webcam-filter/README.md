# CUDA Webcam Filter

## Purpose
This program applies various convolution filters to a webcam feed in real-time using CUDA for GPU acceleration. The application demonstrates how to utilize GPU computing to process video streams efficiently.

## Features
- Real-time webcam video capture
- Multiple convolution filter options (blur, sharpen, edge detection, emboss)
- GPU-accelerated processing using CUDA
- Command-line options for filter selection and parameters

## Usage
```
  cuda-webcam-filter [OPTION...]
```

### List of options
```                            
  -d, --device arg             Camera device ID (default: 0)
  -f, --filter arg             Filter type: blur, sharpen, edge, emboss (default: blur)
  -k, --kernel-size arg        Kernel size for filters (default: 3)
  -s, --sigma arg              Sigma value for Gaussian blur (default: 1.0)
  -i, --intensity arg          Filter intensity (default: 1.0)
  -p, --preview                Show original video alongside filtered
  -h, --help                   Print usage
  -v, --version                Print version information
```

## Hardware requirements
Requires a CUDA-enabled GPU.

## Dependencies
- OpenCV (>= 4.5.0)
- CUDA Toolkit (>= 11.0)
- cxxopts
- plog
- Google Test

## Linux

```
sudo apt-get update
sudo apt-get install libgtk-3-dev pkg-config
```

### External dependencies

The application has the following external dependencies that can be updated using git subtree
- plog - `git subtree pull --prefix templates/cuda-webcam-filter/external/plog https://github.com/SergiusTheBest/plog.git tags/1.1.10 --squash`
- cxxopts - `git subtree pull --prefix templates/cuda-webcam-filter/external/cxxopts https://github.com/jarro2783/cxxopts.git tags/v3.2.0 --squash`
- gtest - `git subtree pull --prefix templates/cuda-webcam-filter/external/gtest https://github.com/google/googletest.git tags/v1.16.0 --squash`
- opencv - `git subtree pull --prefix templates/cuda-webcam-filter/external/opencv https://github.com/opencv/opencv.git tags/4.11.0 --squash`
- opencv contrib - `git subtree pull --prefix templates/cuda-webcam-filter/external/opencv_contrib https://github.com/opencv/opencv_contrib.git tags/4.11.0 --squash`

## Build

### Build on Linux
```bash
mkdir build && cd build
cmake ..
make -j $(nproc)
```

### Build on Windows
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## Testing
The project includes unit tests and functional tests which can be enabled during the build:
```bash
cmake .. -DRUN_UNIT_TESTS=ON
make
cd tests/unit_tests/
ctest
```

## Project Structure
```
cuda-webcam-filter/
├── CMakeLists.txt           # Main build configuration
├── README.md                # Project documentation
├── src/
│   ├── main.cpp             # Application entry point
│   ├── kernels/
│   │   ├── convolution_kernels.cu  # CUDA implementation
│   │   └── kernels.h        # Kernel interfaces
│   ├── utils/
│   │   ├── webcam_handler.cpp  # Camera input/output handling
│   │   ├── webcam_handler.h
│   │   ├── filter_utils.cpp    # Filter creation utilities
│   │   ├── filter_utils.h
│   │   └── version.h.in        # Version template
│   └── input_args_parser/
│       ├── input_args_parser.cpp  # Command line argument parsing
│       └── input_args_parser.h
├── tests/
│   ├── unit_tests/
│   │   ├── CMakeLists.txt
│   │   ├── test_convolution.cpp
│   │   └── test_utils.cpp
│   └── functional_tests/
│       ├── CMakeLists.txt
│       └── test_filters.cpp
└── build/
    └── build_all.sh          # Build script
```

## Example: Adding a New Filter

To add a new filter:

1. Add a new filter type to the `FilterType` enum in `filter_utils.h`
2. Add the filter mapping in `stringToFilterType()` in `filter_utils.cpp`
3. Implement the kernel creation in `createFilterKernel()` in `filter_utils.cpp`

## Performance Considerations

- The provided convolution kernel implementation is optimized for readability but can be further optimized:
  - Consider using shared memory to reduce global memory accesses
  - Explore using texture memory for input images
  - Implement separable convolution for certain filters (like Gaussian blur)
- Profile the application using NVIDIA's profiling tools to identify bottlenecks
