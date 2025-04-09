# Accelerating Application with NVIDIA CUDA C++

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started)


## Examples

### The examples were tested on the following environment
- Ubuntu 24.04.2 LTS
- `nvidia-smi` output
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    Off |   00000000:01:00.0  On |                  N/A |
| N/A   53C    P5             10W /   80W |     112MiB /   8188MiB |      8%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```
- `nvcc --version` output
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```
- `nsys -v` output
```
NVIDIA Nsight Systems version 2025.1.1.131-251135540420v0
```

### Running examples

- open terminal (Bash/x64 Native Tools Command Prompt for VS 2022)
- go to the proper directory, e.g. `cd examples/1-gpu-hello-world`
- compile and run code with nvcc, e.g. `nvcc -o hello.bin hello-world-gpu.cu -run`
- profile application if needed, e.g. `nsys profile --stats=true -o hello-report ./hello.bin` (in case of issues on Windows please set the path to nsys in the command line)
