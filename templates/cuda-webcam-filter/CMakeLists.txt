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

add_subdirectory(external/cxxopts)
add_subdirectory(external/plog)
add_subdirectory(external/gtest)

include_directories(${CMAKE_SOURCE_DIR}/external/cxxopts/include)

# Find required packages
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# CUDA configuration
if(CUDA_SUPPORT)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")

    # Only include CUDA directories if they exist
    if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    else()
        message(WARNING "CUDA toolkit include directories probably not found!")
        # Try to find it manually
        find_path(CUDA_INCLUDE_DIRS cuda_runtime.h
            HINTS /usr/local/cuda/include /usr/include
            PATH_SUFFIXES cuda
        )
        if(CUDA_INCLUDE_DIRS)
            include_directories(${CUDA_INCLUDE_DIRS})
            message(STATUS "Found CUDA includes: ${CUDA_INCLUDE_DIRS}")
        endif()
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
    ${CMAKE_SOURCE_DIR}/src/utils/webcam_handler.cpp
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
    ${OpenCV_LIBS}
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