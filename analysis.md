# Performance Analysis – HDR Tone Mapping

Below is a summary of observed frame‐rates and timings on a **640×480** test image, comparing the CPU and GPU code paths as currently implemented.

| Implementation        | FPS (approx.) | Frame Time (ms) | Notes                                                         |
|-----------------------|---------------|-----------------|---------------------------------------------------------------|
| **CPU-only**          | 3 FPS         | ~155 ms         | `applyHDRToneMappingCPU` on each frame                        |
| **GPU path (stub)**   | 6 FPS         | ~160 ms         | `applyHDRToneMappingGPU` simply calls the CPU fallback        |

---

## Key Observations

1. **Marginal GPU Speedup**  
   Although the GPU path is entered, it delegates back to the CPU routine, so the heavy work remains on the CPU and frame times stay high.

2. **Identified Bottlenecks**  
   - **Per-frame allocations**: repeated `cudaMalloc`/`cudaFree` (formerly) and CPU allocations introduce overhead.  
   - **Synchronous data copies**: blocking `cudaMemcpy` offers no overlap of compute and transfer.  
   - **Lack of real GPU compute**: the CUDA cores and transfer optimizations are unused when the stub is invoked.

---

## Recommendations for True GPU Acceleration

To achieve “real‐time” HDR performance, implement the following:

1. **Parallel CUDA Kernels**  
   - Fuse RGB→luma, tone-mapping, gamma & saturation, and luma→RGB into a single per-pixel kernel to minimize global-memory traffic.

2. **Persistent Allocations**  
   - Allocate device buffers **once** (or only on resolution change).  
   - Use `cudaMallocHost` for pinned host buffers to speed up transfers.

3. **Overlap Transfer & Compute**  
   - Employ `cudaMemcpyAsync` on a CUDA stream, launching the kernel on the same stream, and synchronize once per frame.

4. **Shared Memory for Local Operators**  
   - If adding a local tone-mapping variant (e.g. Gaussian blur on luminance), load tile blocks into shared memory to accelerate neighborhood operations.

---

## Target Performance Goal

- **Achieve ≥ 30 FPS** on 640×480 by leveraging true parallel CUDA execution and optimized memory-transfer strategies.  
- Demonstrate a **5×+ speedup** of the GPU path over the CPU baseline.

With these optimizations, your HDR tone-mapping filter will meet the real-time requirement and fully utilize GPU acceleration. ```
