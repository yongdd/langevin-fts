# CudaDCT - CUDA Discrete Cosine Transform Library

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

CUDA implementation of Discrete Cosine Transform (DCT) Types 1-4, matching FFTW's REDFT conventions. CudaDCT provides GPU-accelerated DCT transforms using cuFFT as the underlying FFT engine, with efficient algorithms that minimize FFT size while maintaining machine precision accuracy.

## Table of Contents

1. [DCT Types and Definitions](#1-dct-types-and-definitions)
2. [FFT-based Implementation](#2-fft-based-implementation)
3. [Multi-dimensional DCT](#3-multi-dimensional-dct)
4. [Normalization](#4-normalization)
5. [API Reference](#5-api-reference)
6. [Usage Examples](#6-usage-examples)
7. [References](#7-references)

---

## 1. DCT Types and Definitions

### 1.1 Overview

The Discrete Cosine Transform (DCT) is a fundamental tool in signal processing and numerical computation, particularly for solving differential equations with various boundary conditions. CudaDCT implements all four standard DCT types matching FFTW's REDFT conventions.

### 1.2 Mathematical Definitions

| Type | FFTW Name | Formula | Grid Points |
|------|-----------|---------|-------------|
| DCT-1 | REDFT00 | $X_k = x_0 + (-1)^k x_{N} + 2\sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi k n}{N}\right)$ | N+1 |
| DCT-2 | REDFT10 | $X_k = 2\sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k (2n+1)}{2N}\right)$ | N |
| DCT-3 | REDFT01 | $X_k = x_0 + 2\sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi n (2k+1)}{2N}\right)$ | N |
| DCT-4 | REDFT11 | $X_k = 2\sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi (2n+1)(2k+1)}{4N}\right)$ | N |

### 1.3 Boundary Conditions and Symmetry

Each DCT type corresponds to specific boundary conditions, which determine their use in solving partial differential equations:

| Type | Left Boundary | Right Boundary | Physical Meaning |
|------|---------------|----------------|------------------|
| DCT-1 | Even at point | Even at point | Neumann-Neumann |
| DCT-2 | Even at half-point | Even at point | Mixed boundary |
| DCT-3 | Even at point | Even at half-point | Mixed boundary |
| DCT-4 | Even at half-point | Even at half-point | Half-sample symmetric |

**Inverse relationships:**
- DCT-1 is self-inverse: $\text{DCT-1}(\text{DCT-1}(x)) = 2N \cdot x$
- DCT-2 and DCT-3 are inverses: $\text{DCT-3}(\text{DCT-2}(x)) = 2N \cdot x$
- DCT-4 is self-inverse: $\text{DCT-4}(\text{DCT-4}(x)) = 2N \cdot x$

---

## 2. FFT-based Implementation

### 2.1 Algorithm Overview

All DCT types are implemented using FFT with pre/post processing, following the cuHelmholtz approach [1]. This achieves optimal computational efficiency by using the minimum FFT size possible.

| DCT Type | FFT Size | FFT Type | Method |
|----------|----------|----------|--------|
| DCT-1 | N | Z2D | cuHelmholtz |
| DCT-2 | N | Z2D | cuHelmholtz |
| DCT-3 | N | D2Z | cuHelmholtz |
| DCT-4 | 2N | Z2Z | Phase rotation |

### 2.2 DCT-1 Algorithm (cuHelmholtz Style)

The DCT-1 transform is computed using an N-point complex-to-real FFT:

**Step 1 - PreOp:**
- Load odd-indexed elements into complex buffer
- Compute differences for reduction
- Parallel reduction for boundary terms

**Step 2 - Z2D FFT:**
- Execute N-point complex-to-real FFT using cuFFT

**Step 3 - PostOp:**
- Reconstruct DCT coefficients using sine-based formula:
$$X_k = \text{Re}(Z_k) + \frac{x_1}{\sin(\pi k / N)}$$

### 2.3 DCT-2 Algorithm (cuHelmholtz Style)

**Step 1 - PreOp:**
- Rearrange input with interleaved indices
- Apply trigonometric weighting

**Step 2 - Z2D FFT:**
- Execute N-point complex-to-real FFT

**Step 3 - PostOp:**
- Extract DCT coefficients with phase correction

### 2.4 DCT-3 Algorithm (cuHelmholtz Style)

DCT-3 is the inverse of DCT-2, using a D2Z (real-to-complex) FFT:

**Step 1 - PreOp:**
- Apply phase factors to input

**Step 2 - D2Z FFT:**
- Execute N-point real-to-complex FFT

**Step 3 - PostOp:**
- Reconstruct output from complex coefficients

### 2.5 DCT-4 Algorithm (Phase Rotation)

DCT-4 requires a 2N-point complex FFT due to the half-sample shift in both input and output:

**Step 1 - PreOp:**
Apply phase rotation and zero-pad to 2N:
$$z_n = x_n \cdot e^{-i\pi(2n+1)/(4N)}, \quad n = 0, \ldots, N-1$$
$$z_n = 0, \quad n = N, \ldots, 2N-1$$

**Step 2 - Z2Z FFT:**
Execute 2N-point complex-to-complex FFT:
$$Z_k = \text{FFT}_{2N}(z)$$

**Step 3 - PostOp:**
Extract DCT-4 coefficients:
$$X_k = 2 \cdot \text{Re}\left(Z_k \cdot e^{-i\pi k/(2N)}\right)$$

---

## 3. Multi-dimensional DCT

### 3.1 Supported Types

All four DCT types are supported for 1D, 2D, and 3D transforms:

| Dimension | Supported Types | Mixed Types | Notes |
|-----------|-----------------|-------------|-------|
| 1D | DCT-1, 2, 3, 4 | N/A | Full support |
| 2D | DCT-1, 2, 3, 4 | DCT-2, 3, 4 | DCT-1 cannot be mixed |
| 3D | DCT-1, 2, 3, 4 | DCT-2, 3, 4 | DCT-1 cannot be mixed |

**Mixed types**: Different DCT types can be applied to different dimensions (e.g., DCT-2 in X, DCT-3 in Y).
DCT-1 cannot be mixed with other types because it uses N+1 points while others use N points.

### 3.2 2D DCT Algorithm

**DCT-1 (2D):**
- Uses mirror extension to convert to DFT
- Single 2D D2Z FFT
- Extract DCT coefficients from symmetric output

**DCT-2 (2D):**
- Batched 1D DCT-2 along Y dimension (preOp → Z2D FFT → postOp)
- Batched 1D DCT-2 along X dimension (preOp → Z2D FFT → postOp)

**DCT-3 (2D):**
- Batched 1D DCT-3 along Y dimension (preOp → D2Z FFT → postOp)
- Transpose to make X contiguous
- Batched 1D DCT-3 along X dimension (preOp → D2Z FFT → postOp with transpose back)

**DCT-4 (2D):**
- Batched 1D DCT-4 along Y dimension (preOp → Z2Z FFT → postOp)
- Batched 1D DCT-4 along X dimension (preOp → Z2Z FFT → postOp)

### 3.3 3D DCT Algorithm

**DCT-1 (3D):**
- Mirror extension in all three dimensions
- Single 3D D2Z FFT
- Extract DCT coefficients

**DCT-2 (3D):**
- Batched 1D DCT-2 along Z dimension (preOp → Z2D FFT → postOp)
- Batched 1D DCT-2 along Y dimension (preOp → Z2D FFT → postOp)
- Batched 1D DCT-2 along X dimension (preOp → Z2D FFT → postOp)

**DCT-3 (3D):**
- Batched 1D DCT-3 along Z dimension (preOp → D2Z FFT → postOp)
- Transpose [Nx][Ny][Nz] → [Nx][Nz][Ny] for Y dimension
- Batched 1D DCT-3 along Y dimension (preOp → D2Z FFT → postOp)
- Transpose [Nx][Ny][Nz] → [Ny][Nz][Nx] for X dimension
- Batched 1D DCT-3 along X dimension (preOp → D2Z FFT → postOp with transpose back)

**DCT-4 (3D):**
- Batched 1D DCT-4 along Z dimension (preOp → Z2Z FFT → postOp)
- Batched 1D DCT-4 along Y dimension (preOp → Z2Z FFT → postOp)
- Batched 1D DCT-4 along X dimension (preOp → Z2Z FFT → postOp)

---

## 4. Normalization

### 4.1 Convention

CudaDCT follows FFTW's unnormalized convention. After a forward-backward (round-trip) transform, the result is scaled by a constant factor.

### 4.2 Normalization Factors

| Transform | Round-trip Scale | Normalization Factor |
|-----------|------------------|---------------------|
| 1D DCT-1 | $2N$ | $1/(2N)$ |
| 1D DCT-2/3 | $2N$ | $1/(2N)$ |
| 1D DCT-4 | $2N$ | $1/(2N)$ |
| 2D (all types) | $4 N_x N_y$ | $1/(4 N_x N_y)$ |
| 3D (all types) | $8 N_x N_y N_z$ | $1/(8 N_x N_y N_z)$ |

Use `get_normalization()` method to obtain the normalization factor programmatically.

---

## 5. API Reference

### 5.1 CudaDCT (1D)

```cpp
class CudaDCT {
public:
    // Constructor: N = transform size, type = DCT type (1-4)
    CudaDCT(int N, CudaDCTType type);
    ~CudaDCT();

    // Execute DCT in-place on device array
    void execute(double* d_data);

    // Get input/output size (N for DCT-2,3,4; N+1 for DCT-1)
    int get_size() const;

    // Get normalization factor for round-trip
    double get_normalization() const;
};
```

**DCT Type Enumeration:**
```cpp
enum CudaDCTType {
    CUDA_DCT_1 = 0,  // FFTW_REDFT00
    CUDA_DCT_2 = 1,  // FFTW_REDFT10
    CUDA_DCT_3 = 2,  // FFTW_REDFT01
    CUDA_DCT_4 = 3   // FFTW_REDFT11
};
```

### 5.2 CudaDCT2D

```cpp
class CudaDCT2D {
public:
    // Constructor: same type for both dimensions (DCT-1,2,3,4)
    CudaDCT2D(int Nx, int Ny, CudaDCTType type);

    // Constructor: different types per dimension (DCT-2,3,4 only)
    // Note: DCT-1 cannot be mixed with other types
    CudaDCT2D(int Nx, int Ny, CudaDCTType type_x, CudaDCTType type_y);
    ~CudaDCT2D();

    // Execute 2D DCT in-place
    void execute(double* d_data);

    // Get total size
    int get_size() const;

    // Get dimensions
    void get_dims(int& nx, int& ny) const;

    // Get DCT types for each dimension
    void get_types(CudaDCTType& type_x, CudaDCTType& type_y) const;

    // Get normalization factor
    double get_normalization() const;
};
```

### 5.3 CudaDCT3D

```cpp
class CudaDCT3D {
public:
    // Constructor: same type for all dimensions (DCT-1,2,3,4)
    CudaDCT3D(int Nx, int Ny, int Nz, CudaDCTType type);

    // Constructor: different types per dimension (DCT-2,3,4 only)
    // Note: DCT-1 cannot be mixed with other types
    CudaDCT3D(int Nx, int Ny, int Nz, CudaDCTType type_x, CudaDCTType type_y, CudaDCTType type_z);
    ~CudaDCT3D();

    // Execute 3D DCT in-place
    void execute(double* d_data);

    // Get total size
    int get_size() const;

    // Get dimensions
    void get_dims(int& nx, int& ny, int& nz) const;

    // Get DCT types for each dimension
    void get_types(CudaDCTType& type_x, CudaDCTType& type_y, CudaDCTType& type_z) const;

    // Get normalization factor
    double get_normalization() const;
};
```

---

## 6. Usage Examples

### 6.1 1D DCT-1

```cpp
#include "CudaDCT.h"

int N = 32;
int size = N + 1;  // DCT-1 has N+1 points

// Allocate and initialize device memory
double* d_data;
cudaMalloc(&d_data, sizeof(double) * size);
cudaMemcpy(d_data, h_input, sizeof(double) * size, cudaMemcpyHostToDevice);

// Create DCT transformer and execute
CudaDCT dct(size, CUDA_DCT_1);
dct.execute(d_data);

// Copy result back
cudaMemcpy(h_output, d_data, sizeof(double) * size, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 6.2 1D DCT-4

```cpp
int N = 64;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * N);
cudaMemcpy(d_data, h_input, sizeof(double) * N, cudaMemcpyHostToDevice);

CudaDCT dct(N, CUDA_DCT_4);
dct.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 6.3 2D DCT-4

```cpp
int Nx = 32, Ny = 64;
int M = Nx * Ny;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * M);
cudaMemcpy(d_data, h_input, sizeof(double) * M, cudaMemcpyHostToDevice);

CudaDCT2D dct(Nx, Ny, CUDA_DCT_4);
dct.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 6.4 3D DCT-4

```cpp
int Nx = 32, Ny = 32, Nz = 32;
int M = Nx * Ny * Nz;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * M);
cudaMemcpy(d_data, h_input, sizeof(double) * M, cudaMemcpyHostToDevice);

CudaDCT3D dct(Nx, Ny, Nz, CUDA_DCT_4);
dct.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 6.5 2D Mixed DCT (DCT-2 in X, DCT-3 in Y)

```cpp
int Nx = 32, Ny = 64;
int M = Nx * Ny;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * M);
cudaMemcpy(d_data, h_input, sizeof(double) * M, cudaMemcpyHostToDevice);

// Different DCT types per dimension
CudaDCT2D dct(Nx, Ny, CUDA_DCT_2, CUDA_DCT_3);
dct.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 6.6 3D Mixed DCT

```cpp
int Nx = 32, Ny = 32, Nz = 32;
int M = Nx * Ny * Nz;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * M);
cudaMemcpy(d_data, h_input, sizeof(double) * M, cudaMemcpyHostToDevice);

// DCT-2 in X, DCT-3 in Y, DCT-4 in Z
CudaDCT3D dct(Nx, Ny, Nz, CUDA_DCT_2, CUDA_DCT_3, CUDA_DCT_4);
dct.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 6.7 Round-trip with Normalization

```cpp
int N = 64;
double* d_data;
cudaMalloc(&d_data, sizeof(double) * N);
cudaMemcpy(d_data, h_input, sizeof(double) * N, cudaMemcpyHostToDevice);

CudaDCT dct(N, CUDA_DCT_4);

// Forward transform
dct.execute(d_data);

// Backward transform (DCT-4 is self-inverse)
dct.execute(d_data);

// Apply normalization to recover original
double norm = dct.get_normalization();
// Scale d_data by norm on GPU...

cudaMemcpy(h_output, d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
cudaFree(d_data);
// h_output should equal h_input
```

---

## 7. References

1. Ren, M., Gao, Y., Wang, G., & Liu, X. (2020). "Discrete Sine and Cosine Transform and Helmholtz Equation Solver on GPU." *2020 IEEE ISPA/BDCloud/SocialCom/SustainCom*, 57-66. DOI: [10.1109/ISPA-BDCloud-SocialCom-SustainCom51426.2020.00034](https://doi.org/10.1109/ISPA-BDCloud-SocialCom-SustainCom51426.2020.00034)

2. cuHelmholtz GitHub Repository: https://github.com/rmingming/cuHelmholtz

3. FFTW Manual - Real-even DFTs (Discrete Cosine Transforms): http://www.fftw.org/fftw3_doc/Real-even_002fodd-DFTs-_0028Cosine_002fSine-Transforms_0029.html

4. Makhoul, J. (1980). "A Fast Cosine Transform in One and Two Dimensions." *IEEE Trans. Acoust., Speech, Signal Process.*, 28(1), 27-34.

5. Britanak, V., Yip, P., & Rao, K.R. (2007). *Discrete Cosine and Sine Transforms: General Properties, Fast Algorithms and Integer Approximations.* Academic Press.
