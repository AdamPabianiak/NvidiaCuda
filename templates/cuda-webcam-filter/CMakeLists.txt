cmake_minimum_required(VERSION 3.27) # Requires 25 or above to have CUDA20 standard flag
project(cuda-webcam-filter LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE OFF)

# Project version
set(PROJECT_VERSION "0.1.0")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/src/utils/version.h
    @ONLY
)

# Options
option(CUDA_SUPPORT "Enable CUDA support" ON)
option(CUDA_DEBUG "Enable CUDA debug" OFF)
option(RUN_UNIT_TESTS "Enable running unit tests" OFF)
option(RUN_FUNCTIONAL_TESTS "Enable running functional tests" OFF)

# Add dependencies before OpenCV
add_subdirectory(external/cxxopts)
add_subdirectory(external/plog)
add_subdirectory(external/gtest)

include_directories(${CMAKE_SOURCE_DIR}/external/cxxopts/include)

# Set path to opencv_contrib modules - this is critical for CUDA support
set(OPENCV_EXTRA_MODULES_PATH "${CMAKE_SOURCE_DIR}/external/opencv_contrib/modules" CACHE PATH "Location of opencv_contrib modules")

# Common OpenCV configuration options
set(WITH_OPENGL ON CACHE BOOL "")
set(WITH_VTK OFF CACHE BOOL "")
set(WITH_QT OFF CACHE BOOL "")
set(BUILD_opencv_python2 OFF CACHE BOOL "")
set(BUILD_opencv_python3 OFF CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_PERF_TESTS OFF CACHE BOOL "")
set(BUILD_DOCS OFF CACHE BOOL "")
set(BUILD_JAVA OFF CACHE BOOL "")

# OS-specific configurations
if(WIN32)
    # Windows-specific settings
    message(STATUS "Configuring build for Windows")
    set(WITH_GTK OFF CACHE BOOL "")
    set(WITH_GTK_2_X OFF CACHE BOOL "")
    
    # Windows-specific CUDA paths
    if(CUDA_SUPPORT)
        # Try to find CUDA in standard locations
        find_path(CUDA_INCLUDE_DIRS cuda_runtime.h
            HINTS 
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.7/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include"
            PATH_SUFFIXES cuda
        )
        if(CUDA_INCLUDE_DIRS)
            include_directories(${CUDA_INCLUDE_DIRS})
            message(STATUS "Found CUDA includes: ${CUDA_INCLUDE_DIRS}")
        endif()
    endif()
    
    # Use MSVS-specific settings if using Visual Studio
    if(MSVC)
        add_compile_options(/MP) # Enable parallel builds
    endif()
else()
    # Linux-specific settings
    message(STATUS "Configuring build for Linux")
    set(WITH_GTK ON CACHE BOOL "")
    set(WITH_GTK_2_X OFF CACHE BOOL "") # Use GTK3 by default on Linux
    
    # Linux-specific CUDA paths
    if(CUDA_SUPPORT)
        find_path(CUDA_INCLUDE_DIRS cuda_runtime.h
            HINTS 
            /usr/local/cuda/include
            /usr/include
            /opt/cuda/include
            PATH_SUFFIXES cuda
        )
        if(CUDA_INCLUDE_DIRS)
            include_directories(${CUDA_INCLUDE_DIRS})
            message(STATUS "Found CUDA includes: ${CUDA_INCLUDE_DIRS}")
        endif()
    endif()
endif()

# CUDA support for OpenCV
if(CUDA_SUPPORT)
    set(WITH_CUDA ON CACHE BOOL "")
    set(WITH_CUBLAS ON CACHE BOOL "")
    set(CUDA_FAST_MATH ON CACHE BOOL "")
    set(OPENCV_DNN_CUDA OFF CACHE BOOL "")
    
    # Disable certain CUDA features that might cause problems
    set(WITH_CUDNN OFF CACHE BOOL "")
    
    # Build cudev module that is required for CUDA support
    set(BUILD_opencv_cudev ON CACHE BOOL "")
else()
    set(WITH_CUDA OFF CACHE BOOL "")
    set(WITH_CUBLAS OFF CACHE BOOL "")
endif()

# Add OpenCV subdirectory
add_subdirectory(external/opencv)

# Explicitly include OpenCV directories - both source and build directories
include_directories(
    ${CMAKE_SOURCE_DIR}/external/opencv/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/core/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/calib3d/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/features2d/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/flann/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/dnn/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/highgui/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/imgcodecs/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/videoio/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/imgproc/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/ml/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/objdetect/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/photo/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/stitching/include
    ${CMAKE_SOURCE_DIR}/external/opencv/modules/video/include
    ${CMAKE_BINARY_DIR}/external/opencv
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/modules
)

# Also include contrib modules if CUDA is enabled
if(CUDA_SUPPORT)
    include_directories(
        ${CMAKE_SOURCE_DIR}/external/opencv_contrib/modules/cudev/include
        ${CMAKE_BINARY_DIR}/opencv_modules
        ${CMAKE_BINARY_DIR}/external/opencv/modules # For generated headers
    )
endif()

# CUDA configuration
if(CUDA_SUPPORT)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")

    # Include CUDA toolkit directories
    if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    endif()
    
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++20")
    
    # Target various CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4060 (Ada Lovelace) - Adjust to target specific GPUs!!!
    
    if(CUDA_DEBUG)
        message(STATUS "CUDA Debug is ON")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G") # enable cuda-gdb
    endif()

    message(STATUS "CMAKE_CUDA_FLAGS = ${CMAKE_CUDA_FLAGS}")
endif()

# Define source files
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/input_handler.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/filter_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/input_args_parser/input_args_parser.cpp
)

# CUDA specific files
if(CUDA_SUPPORT)
    list(APPEND SOURCES
        ${CMAKE_SOURCE_DIR}/src/kernels/convolution_kernels.cu
    )
    # Treat main.cpp as CUDA file
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/main.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Include directories
set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/kernels/
    ${CMAKE_SOURCE_DIR}/src/utils/
    ${CMAKE_SOURCE_DIR}/src/input_args_parser/
    ${CMAKE_CURRENT_BINARY_DIR}/src/utils/
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    plog::plog
    GTest::gtest GTest::gmock
)

# Testing
if(RUN_UNIT_TESTS)
    add_subdirectory(tests/unit_tests)
endif()

# Not implemented
# if(RUN_FUNCTIONAL_TESTS)
#     add_subdirectory(tests/functional_tests)
# endif()

if((NOT RUN_UNIT_TESTS) AND (NOT RUN_FUNCTIONAL_TESTS))
    add_custom_target(run_tests
        COMMAND echo "Running tests skipped...")
endif()