# CUDA Matrix Transpose – Assignment Report

> **Author**: Adam  
> **GPU / Driver / CUDA**: GeForce GTX 1070 Ti (SM 6.1 Pascal), driver 555.xx, CUDA 12.9  
> **Host compiler**: MSVC 19.38 (VS 2022)  
> **Date**: 2025-06-23

---

## 1  Overview
This project implements and benchmarks three versions of a matrix-transpose routine:

| Version           | Memory model                   | Main idea                                                               |
|-------------------|--------------------------------|-------------------------------------------------------------------------|
| **CPU reference** | Host RAM                       | Simple nested `for` loops                                               |
| **GPU naïve**     | Global memory                  | 1 thread ⇢ 1 element (no tiling)                                       |
| **GPU optimised** | Unified Memory + Shared Memory | 16×16 / 32×32 padded tiles, coalesced loads/stores, `__shared__` buffer |

We begin with a **2048 × 1024** matrix and scale to **8192 × 8192**.  All kernels were compiled with `nvcc -std=c++17 -O3` and run on the same GPU/driver session.

---

## 2  How to run it quickly (Windows & Linux)

```bash
# 1  Clone / unzip the repo
> git clone <your-repo> && cd cuda-transpose

# 2  Build both executables
#    Windows – open the *x64 Native Tools Command Prompt for VS 2022*
> nvcc -std=c++17 -O3 transpose_part1.cu -o transpose_part1.exe
> nvcc -std=c++17 -O3 transpose_part2.cu -o transpose_part2.exe

#    Linux/WSL – any Bash shell with CUDA 12.9 toolkit
$ nvcc -std=c++17 -O3 transpose_part1.cu -o transpose_part1
$ nvcc -std=c++17 -O3 transpose_part2.cu -o transpose_part2

# 3  Run ( N  M  TILE )
> transpose_part1.exe              # default 2048×1024, tile16
> transpose_part2.exe 4096 4096 32 # big matrix, optimised kernel, 32×32 tile
```
*No CMake needed – each file is single-translation-unit.*

---

## 3  Execution-time summary

### 3.1  2048 × 1024 (Part 1 only)
| Impl | Tile   | Time (ms) | GB/s  | Speed-up vs CPU | Correct |
|------|-----:  |----------:|-----: |----------------:|:-------:|
| **CPU** | –   | 9.422     | 1.78  | 1×              | ✅ |
| **GPU naïve** | 16        | 0.355 | 47.3 | 26.6×    | ✅ |

### 3.2  4096 × 4096
| Impl | Tile            | Time (ms)    | GB/s  | Speed-up vs CPU        | Correct |
|------|-----:|----------:       |-----:|----------------:|:-------:     |
| CPU | –                | 150.76       | 0.89  | 1×                     | ✅ |
| GPU naïve (global mem) | 16           | 1.568 | 85.6    | 96×          | ✅ |
| GPU naïve **UM**       | 16           | 39.25 | 3.4     | 3.9×         | ✅ |
| GPU **optimised** UM   | 16           | 0.661 | 203     | 228×         | ✅ |
| GPU **optimised** UM   | 32           | 0.671 | 200     | 225×         | ✅ |

### 3.3  8192 × 8192 (Unified Memory)
| Impl | Tile      | Time (ms) | GB/s   | Speed-up vs CPU | Correct |
|------|-----:     |----------:|-----:  |----------------:|:-------:|
| CPU | –          | 712.78    | 0.75   | 1×              | ✅ |
| GPU naïve UM     | 16        | 143.44 | 3.7 | 5.0×      | ✅ |
| GPU optimised UM | 16        | 3.887  | 138 | 183×      | ✅ |
| GPU optimised UM | 32        | 2.704  | 199 | 263×      | ✅ |

---

## 4  Performance analysis

### 4.1  Why naïve GPU already beats CPU
* **Massive parallelism** – thousands of CUDA cores vs one CPU thread.
* **Memory concurrency** – GPU can issue >100 outstanding reads; CPU loop is cache-latency bound.

### 4.2  Unified Memory first-touch penalty
The Part-2 program uses Unified Memory for convenience.  The **first launch** of the naïve kernel incurs on-demand page migrations across PCIe, throttling bandwidth to ~3 GB/s.  A second launch (not shown) runs near the Part-1 numbers once pages are resident.

### 4.3  Shared-memory tiling benefits
1. **Coalesced reads/writes** – threads in a warp access consecutive addresses.
2. **Bank-conflict padding** (`+1` column) – avoids 32-way shared-mem conflicts during transpose.
3. **Reduced global transactions** – each element is loaded & stored exactly once.

Result: ~2.3× speed-up over naïve global-memory kernel and >200 GB/s sustained bandwidth (≈60 % of this Turing GPU’s theoretical 336 GB/s).

### 4.4  Impact of matrix size
* CPU runtime grows quadratically with N; bandwidth stays flat once data exceeds LLC.
* Optimised GPU kernel is **memory-bound**; doubling each dimension (×4 elements) multiplies time by ≈4 while GB/s remains nearly constant (~200 GB/s).
* Naïve UM kernel scales poorly because PCIe migrations grow with data size.

---

## 5  Further optimisation ideas
| Idea                                               | Expected gain            | Notes |
|------                                              |--------------            |-------|
| **Prefetch UM pages** with `cudaMemPrefetchAsync`  | Removes first-touch cost | trivial code change |
| Use **explicit `cudaMemcpy`** instead of UM        | ~12× vs naïve UM         | matches Part-1 naïve timing |
| **Vectorised `float4` LD/ST**                      | +15 % BW                 | halves memory transactions |
| Warp-level **`ldmatrix` transpose** (Ampere+)      | ~1.3×                    | avoids shared memory entirely |
| **Asynchronous copy** (`cp.async`) + double-buffer | overlaps loads & stores  | requires SM 8.6+ |

---

## 6  Project structure
```
│  README.md          ← this report
│  transpose_part1.cu ← CPU + naïve GPU kernel (global memory)
│  transpose_part2.cu ← Unified Memory + shared-memory tiled kernel
│  benchmark.sh       ← build & run sweep, writes results.csv  (optional)
│  analyze.py         ← loads CSV, prints table, saves bandwidth.png
```