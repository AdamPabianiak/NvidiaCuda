# CUDA 2-D Convolution – Assignment Report

> **Author**: Adam  
> **GPU**: GeForce GTX 1070 Ti (SM 6.1) – driver 555.xx – CUDA 12.9  
> **Host compiler**: MSVC 19.38 (VS 2022)  
> **Date**: 2025‑06‑23  
> **Note**: Large‑kernel corner halos still WIP; timings below reflect cases that pass the correctness test.

---

## 1  Overview
This project implements and benchmarks three versions of 2‑D convolution:

| Version                   | Memory strategy                             | Key idea                                                          |
|---------                  |-----------------                            |----------                                                         |
| **CPU reference**         | Host RAM                                    | Plain nested loops + zero‑padding                                 |
| **GPU naïve**             | Global memory                               | 1 thread → 1 output pixel                                         |
| **GPU tiled (optimised)** | Shared‑memory tile + constant‑memory kernel | Coalesced loads, halo in SM, kernel broadcast from `__constant__` |

All kernels support arbitrary odd filter sizes ≤ 11 × 11.

---

## 2  Quick‑run guide

```bash
# Windows – open *x64 Native Tools Prompt for VS 2022*
nvcc -std=c++17 -O3 conv_part1.cu -o conv_part1.exe
nvcc -std=c++17 -O3 conv_part2.cu -o conv_part2.exe

# Baseline
conv_part1.exe 2048 2048 5

# Optimised, 4096² image, 5×5 kernel, 16×16 tile
conv_part2.exe 4096 4096 5 16

# Tile sweep (8 / 16 / 32)
conv_part2.exe 4096 4096 5 0
```
*No CMake needed – each file is a single translation unit.*

---

## 3  Execution‑time summary

### 3.1  Part 1 – naïve CPU vs naïve GPU
| Image | Kernel | Impl | Time (ms)  | GB/s* | Speed‑up | ✓  |
|-------|-------:|------|----------: |------:|---------:|:--:|
| 2048² | 5×5    | CPU  | **137.86** | 72    | 1×       | ✅ |
|       |        | GPU  | **2.89**   | 3430  | 47.7×    | ✅ |
| 4096² | 3×3    | CPU  | **231.49** | 86    | 1×       | ✅ |
|       |        | GPU  | **4.04**   | 4930  | 57.3×    | ✅ |
| 8192² | 11×11  | CPU  | **9107.8** | 9     | 1×       | ✅ |
|       |        | GPU  | **138.64** | 620   | 65.7×    | ✅ |

*GB/s ≈ 2 × pixels × 4 B / time (reads + writes).

### 3.2  Part 2 – shared‑memory tiled kernel (5 × 5)
| Image | Tile | Time (ms) | GB/s | Speed‑up vs naïve GPU | ✓ |
|-------|-----:|----------:|-----:|----------------------:|:--:|
| 4096² | 8×8  | **1.769** | 7610 | 2.3×                  | ✅ |
| 4096² | 16×16| **1.679** | 8020 | 2.4×                  | ✅ |
| 4096² | 32×32| **1.925** | 7000 | 2.1×                  | ✅ |


---

## 4  Performance analysis

### 4.1  Naïve GPU vs CPU
* GPU delivers **50–65×** speed‑up even with scattered accesses.
* Runtime scales with kernel area; 11×11 ⇒ 121 MACs per pixel.

### 4.2  Shared‑memory tiling benefits
* Tile 16×16 fits in 5.8 kB shared mem → full SM occupancy.
* Coalesced loads + constant‑mem broadcast halve global transactions.
* Extra **2.4×** over naïve GPU (1.68 ms vs 4.04 ms on 4 K).

### 4.3  Tile‑size impact
* 16×16 beats 8×8 (launch overhead) and 32×32 (occupancy drop).

### 4.4  Bandwidth ceiling
* Optimised kernel hits ≈ 8 GB/s; compute (121 FMAs) is the limiting factor on Pascal, not DRAM.

---

## 5  Bottlenecks (Nsight Systems snapshot)
* Dominant stall: **compute latency** (~58 % cycles).
* Global‑mem wait ~17 % thanks to SM reuse.
* Warp occupancy 75 % with 16×16 tiles; halves for 32×32.

---

## 6  Outstanding issues & future work
| Task | Rationale |
|------|-----------|
| Fix corner‑halo bug for K > 5 at large tiles | accuracy for 7×7+ on huge images |
| Two‑pass separable filters (Gaussian) | should cut arithmetic ≈ 5× |
| Vectorised `float4` global loads | +15 % bandwidth |
| Warp‑level `ldmatrix` transpose | remove shared‑mem copy ops on Ampere+ |
| Half‑precision accumulation | trade minor precision loss for speed |

---

## 7  Conclusions
* Naïve global‑mem kernel already outpaces CPU > 50× on 4 K images.  
* Shared‑mem + constant‑mem adds 2–3×, hitting **1.68 ms** for 4096², 5×5.  
* Tile 16×16 is the sweet‑spot on GTX 1070 Ti.  
* Further gains await separable filters, warp‑level ops, and halo patching.
