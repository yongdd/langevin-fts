# CudaDCT / CudaDST - CUDA Discrete Cosine and Sine Transform Library

> **Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

CUDA implementation of Discrete Cosine Transform (DCT) and Discrete Sine Transform (DST) Types 1-4, matching FFTW's REDFT/RODFT conventions. These libraries provide GPU-accelerated transforms using cuFFT as the underlying FFT engine, with efficient algorithms that minimize FFT size while maintaining machine precision accuracy.

## Table of Contents

1. [DCT Types and Definitions](#1-dct-types-and-definitions)
2. [DST Types and Definitions](#2-dst-types-and-definitions)
3. [FFT-based Implementation](#3-fft-based-implementation)
4. [Multi-dimensional Transforms](#4-multi-dimensional-transforms)
5. [Normalization](#5-normalization)
6. [API Reference](#6-api-reference)
7. [Usage Examples](#7-usage-examples)
8. [References](#8-references)

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

## 2. DST Types and Definitions

### 2.1 Overview

The Discrete Sine Transform (DST) is the counterpart to the DCT, used for solving differential equations with Dirichlet (zero) boundary conditions. CudaDST implements all four standard DST types matching FFTW's RODFT conventions.

### 2.2 Mathematical Definitions

| Type | FFTW Name | Formula | Grid Points |
|------|-----------|---------|-------------|
| DST-1 | RODFT00 | $Y_k = 2\sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (n+1)(k+1)}{N+1}\right)$ | N |
| DST-2 | RODFT10 | $Y_k = 2\sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (2n+1)(k+1)}{2N}\right)$ | N |
| DST-3 | RODFT01 | $Y_k = (-1)^k x_{N-1} + 2\sum_{n=0}^{N-2} x_n \sin\left(\frac{\pi (n+1)(2k+1)}{2N}\right)$ | N |
| DST-4 | RODFT11 | $Y_k = 2\sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (2n+1)(2k+1)}{4N}\right)$ | N |

### 2.3 Boundary Conditions and Symmetry

Each DST type corresponds to specific boundary conditions with odd (anti-)symmetry:

| Type | Left Boundary | Right Boundary | Physical Meaning |
|------|---------------|----------------|------------------|
| DST-1 | Odd at point | Odd at point | Dirichlet-Dirichlet |
| DST-2 | Odd at half-point | Odd at point | Mixed boundary |
| DST-3 | Odd at point | Odd at half-point | Mixed boundary |
| DST-4 | Odd at half-point | Odd at half-point | Half-sample antisymmetric |

**Inverse relationships:**
- DST-1 is self-inverse: $\text{DST-1}(\text{DST-1}(x)) = 2(N+1) \cdot x$
- DST-2 and DST-3 are inverses: $\text{DST-3}(\text{DST-2}(x)) = 2N \cdot x$
- DST-4 is self-inverse: $\text{DST-4}(\text{DST-4}(x)) = 2N \cdot x$

---

## 3. FFT-based Implementation

### 3.1 DCT Algorithm Overview

All DCT types are implemented using FFT with pre/post processing, following the cuHelmholtz approach [1]. This achieves optimal computational efficiency by using the minimum FFT size possible.

| DCT Type | FFT Size | FFT Type | Method |
|----------|----------|----------|--------|
| DCT-1 | N | Z2D | cuHelmholtz |
| DCT-2 | N | Z2D | cuHelmholtz |
| DCT-3 | N | D2Z | cuHelmholtz |
| DCT-4 | 2N | Z2Z | Phase rotation |

### 3.2 DST Algorithm Overview

DST types are implemented using FFT with explicit odd-symmetric extension. The extension creates an antisymmetric signal that can be processed via standard FFT.

| DST Type | FFT Size | FFT Type | Method |
|----------|----------|----------|--------|
| DST-1 | 2(N+1) | D2Z | Odd extension |
| DST-2 | 4N | D2Z | Odd extension |
| DST-3 | 4N | D2Z | Odd extension with even reflection |
| DST-4 | 2N | Z2Z | Phase rotation |

### 3.3 DCT-1 Algorithm (cuHelmholtz Style)

**Step 1 - PreOp:**
- Load odd-indexed elements into complex buffer
- Compute differences for reduction
- Parallel reduction for boundary terms

**Step 2 - Z2D FFT:**
- Execute N-point complex-to-real FFT using cuFFT

**Step 3 - PostOp:**
- Reconstruct DCT coefficients using sine-based formula:
$$X_k = \text{Re}(Z_k) + \frac{x_1}{\sin(\pi k / N)}$$

### 3.4 DCT-4 Algorithm (Phase Rotation)

DCT-4 requires a 2N-point complex FFT due to the half-sample shift:

**Step 1 - PreOp:**
$$z_n = x_n \cdot e^{-i\pi(2n+1)/(4N)}, \quad n = 0, \ldots, N-1$$
$$z_n = 0, \quad n = N, \ldots, 2N-1$$

**Step 2 - Z2Z FFT:**
$$Z_k = \text{FFT}_{2N}(z)$$

**Step 3 - PostOp:**
$$X_k = 2 \cdot \text{Re}\left(Z_k \cdot e^{-i\pi k/(2N)}\right)$$

### 3.5 DST-1 Algorithm (Odd Extension)

DST-1 uses a 2(N+1)-size D2Z FFT with odd extension:

**Step 1 - PreOp:**
Construct odd-symmetric extension:
```
work[0] = 0
work[j] = x[j-1]       for j = 1..N
work[N+1] = 0
work[2(N+1)-j] = -x[j-1]  for j = 1..N
```

**Step 2 - D2Z FFT:**
Execute 2(N+1)-point real-to-complex FFT

**Step 3 - PostOp:**
$$Y_k = -\text{Im}(\text{FFT}[k+1])$$

### 3.6 DST-2 Algorithm (Odd Extension)

DST-2 uses a 4N-size D2Z FFT:

**Step 1 - PreOp:**
```
work[2j+1] = x[j]      for j = 0..N-1
work[4N-2j-1] = -x[j]  for j = 0..N-1
all other elements = 0
```

**Step 2 - D2Z FFT:**
Execute 4N-point real-to-complex FFT

**Step 3 - PostOp:**
$$Y_k = -\text{Im}(\text{FFT}[k+1])$$

### 3.7 DST-3 Algorithm (Odd Extension with Even Reflection)

DST-3 uses a 4N-size D2Z FFT with combined even reflection and odd extension:

**Step 1 - PreOp:**
```
work[0] = 0
work[j] = x[j-1]           for j = 1..N
work[2N-j] = x[j-1]        for j = 1..N-1  (even reflection)
work[2N] = 0
work[2N+j] = -work[2N-j]   for j = 1..2N-1 (odd extension)
```

**Step 2 - D2Z FFT:**
Execute 4N-point real-to-complex FFT

**Step 3 - PostOp:**
$$Y_k = -0.5 \cdot \text{Im}(\text{FFT}[2k+1])$$

### 3.8 DST-4 Algorithm (Phase Rotation)

DST-4 uses a 2N-point complex FFT analogous to DCT-4:

**Step 1 - PreOp:**
$$z_n = x_n \cdot e^{-i\pi(2n+1)/(4N)}, \quad n = 0, \ldots, N-1$$
$$z_n = 0, \quad n = N, \ldots, 2N-1$$

**Step 2 - Z2Z FFT:**
$$Z_k = \text{FFT}_{2N}(z)$$

**Step 3 - PostOp:**
$$Y_k = -2 \cdot \text{Im}\left(Z_k \cdot e^{-i\pi k/(2N)}\right)$$

Note: The only difference from DCT-4 is that DST-4 extracts the imaginary part (with sign flip) instead of the real part.

---

## 4. Multi-dimensional Transforms

### 4.1 Supported Types

All four DCT and DST types are supported for 1D, 2D, and 3D transforms:

| Dimension | Supported Types | Mixed Types | Notes |
|-----------|-----------------|-------------|-------|
| 1D | DCT/DST-1,2,3,4 | N/A | Full support |
| 2D | DCT/DST-1,2,3,4 | DCT/DST-2,3,4 | Type-1 cannot be mixed |
| 3D | DCT/DST-1,2,3,4 | DCT/DST-2,3,4 | Type-1 cannot be mixed |

**Mixed types**: Different transform types can be applied to different dimensions (e.g., DST-2 in X, DST-3 in Y).
Type-1 transforms cannot be mixed with other types because they use different point counts (N+1 vs N for DCT, N vs N for DST but different normalization).

### 4.2 2D Transform Algorithm

**Type-1 (2D):**
- Uses mirror/odd extension to convert to DFT
- Single 2D D2Z FFT
- Extract coefficients from symmetric output

**Types 2,3,4 (2D):**
- Batched 1D transforms along Y dimension (preOp → FFT → postOp)
- Batched 1D transforms along X dimension (preOp → FFT → postOp)

### 4.3 3D Transform Algorithm

**Type-1 (3D):**
- Extension in all three dimensions
- Single 3D D2Z FFT
- Extract coefficients

**Types 2,3,4 (3D):**
- Batched 1D transforms along Z dimension (preOp → FFT → postOp)
- Batched 1D transforms along Y dimension (preOp → FFT → postOp)
- Batched 1D transforms along X dimension (preOp → FFT → postOp)

---

## 5. Normalization

### 5.1 Convention

Both CudaDCT and CudaDST follow FFTW's unnormalized convention. After a forward-backward (round-trip) transform, the result is scaled by a constant factor.

### 5.2 DCT Normalization Factors

| Transform | Round-trip Scale | Normalization Factor |
|-----------|------------------|---------------------|
| 1D DCT-1 | $2N$ | $1/(2N)$ |
| 1D DCT-2/3 | $2N$ | $1/(2N)$ |
| 1D DCT-4 | $2N$ | $1/(2N)$ |
| 2D (all types) | $4 N_x N_y$ | $1/(4 N_x N_y)$ |
| 3D (all types) | $8 N_x N_y N_z$ | $1/(8 N_x N_y N_z)$ |

### 5.3 DST Normalization Factors

| Transform | Round-trip Scale | Normalization Factor |
|-----------|------------------|---------------------|
| 1D DST-1 | $2(N+1)$ | $1/(2(N+1))$ |
| 1D DST-2/3 | $2N$ | $1/(2N)$ |
| 1D DST-4 | $2N$ | $1/(2N)$ |
| 2D DST-1 | $4(N_x+1)(N_y+1)$ | $1/(4(N_x+1)(N_y+1))$ |
| 2D DST-2/3/4 | $4 N_x N_y$ | $1/(4 N_x N_y)$ |
| 3D DST-1 | $8(N_x+1)(N_y+1)(N_z+1)$ | $1/(8(N_x+1)(N_y+1)(N_z+1))$ |
| 3D DST-2/3/4 | $8 N_x N_y N_z$ | $1/(8 N_x N_y N_z)$ |

Use `get_normalization()` method to obtain the normalization factor programmatically.

---

## 6. API Reference

### 6.1 CudaDCT (1D)

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

### 6.2 CudaDST (1D)

```cpp
class CudaDST {
public:
    // Constructor: N = transform size, type = DST type (1-4)
    CudaDST(int N, CudaDSTType type);
    ~CudaDST();

    // Execute DST in-place on device array
    void execute(double* d_data);

    // Get input/output size
    int get_size() const;

    // Get normalization factor for round-trip
    double get_normalization() const;
};
```

**DST Type Enumeration:**
```cpp
enum CudaDSTType {
    CUDA_DST_1 = 0,  // FFTW_RODFT00
    CUDA_DST_2 = 1,  // FFTW_RODFT10
    CUDA_DST_3 = 2,  // FFTW_RODFT01
    CUDA_DST_4 = 3   // FFTW_RODFT11
};
```

### 6.3 CudaDCT2D / CudaDST2D

```cpp
class CudaDCT2D {
public:
    // Constructor: same type for both dimensions
    CudaDCT2D(int Nx, int Ny, CudaDCTType type);

    // Constructor: different types per dimension (types 2,3,4 only)
    CudaDCT2D(int Nx, int Ny, CudaDCTType type_x, CudaDCTType type_y);
    ~CudaDCT2D();

    void execute(double* d_data);
    int get_size() const;
    void get_dims(int& nx, int& ny) const;
    void get_types(CudaDCTType& type_x, CudaDCTType& type_y) const;
    double get_normalization() const;
};

class CudaDST2D {
public:
    // Constructor: same type for both dimensions
    CudaDST2D(int Nx, int Ny, CudaDSTType type);

    // Constructor: different types per dimension (types 2,3,4 only)
    CudaDST2D(int Nx, int Ny, CudaDSTType type_x, CudaDSTType type_y);
    ~CudaDST2D();

    void execute(double* d_data);
    int get_size() const;
    void get_dims(int& nx, int& ny) const;
    void get_types(CudaDSTType& type_x, CudaDSTType& type_y) const;
    double get_normalization() const;
};
```

### 6.4 CudaDCT3D / CudaDST3D

```cpp
class CudaDCT3D {
public:
    // Constructor: same type for all dimensions
    CudaDCT3D(int Nx, int Ny, int Nz, CudaDCTType type);

    // Constructor: different types per dimension (types 2,3,4 only)
    CudaDCT3D(int Nx, int Ny, int Nz, CudaDCTType type_x, CudaDCTType type_y, CudaDCTType type_z);
    ~CudaDCT3D();

    void execute(double* d_data);
    int get_size() const;
    void get_dims(int& nx, int& ny, int& nz) const;
    void get_types(CudaDCTType& type_x, CudaDCTType& type_y, CudaDCTType& type_z) const;
    double get_normalization() const;
};

class CudaDST3D {
public:
    // Constructor: same type for all dimensions
    CudaDST3D(int Nx, int Ny, int Nz, CudaDSTType type);

    // Constructor: different types per dimension (types 2,3,4 only)
    CudaDST3D(int Nx, int Ny, int Nz, CudaDSTType type_x, CudaDSTType type_y, CudaDSTType type_z);
    ~CudaDST3D();

    void execute(double* d_data);
    int get_size() const;
    void get_dims(int& nx, int& ny, int& nz) const;
    void get_types(CudaDSTType& type_x, CudaDSTType& type_y, CudaDSTType& type_z) const;
    double get_normalization() const;
};
```

---

## 7. Usage Examples

### 7.1 1D DCT-4

```cpp
#include "CudaRealTransform.h"

int N = 64;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * N);
cudaMemcpy(d_data, h_input, sizeof(double) * N, cudaMemcpyHostToDevice);

CudaDCT dct(N, CUDA_DCT_4);
dct.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 7.2 1D DST-4

```cpp
#include "CudaRealTransform.h"

int N = 64;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * N);
cudaMemcpy(d_data, h_input, sizeof(double) * N, cudaMemcpyHostToDevice);

CudaDST dst(N, CUDA_DST_4);
dst.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 7.3 2D DST-1

```cpp
int Nx = 32, Ny = 64;
int M = Nx * Ny;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * M);
cudaMemcpy(d_data, h_input, sizeof(double) * M, cudaMemcpyHostToDevice);

CudaDST2D dst(Nx, Ny, CUDA_DST_1);
dst.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 7.4 3D DST-4

```cpp
int Nx = 32, Ny = 32, Nz = 32;
int M = Nx * Ny * Nz;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * M);
cudaMemcpy(d_data, h_input, sizeof(double) * M, cudaMemcpyHostToDevice);

CudaDST3D dst(Nx, Ny, Nz, CUDA_DST_4);
dst.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 7.5 2D Mixed DST (DST-2 in X, DST-3 in Y)

```cpp
int Nx = 32, Ny = 64;
int M = Nx * Ny;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * M);
cudaMemcpy(d_data, h_input, sizeof(double) * M, cudaMemcpyHostToDevice);

// Different DST types per dimension
CudaDST2D dst(Nx, Ny, CUDA_DST_2, CUDA_DST_3);
dst.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 7.6 3D Mixed DCT

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

### 7.7 Round-trip with Normalization

```cpp
int N = 64;
double* d_data;
cudaMalloc(&d_data, sizeof(double) * N);
cudaMemcpy(d_data, h_input, sizeof(double) * N, cudaMemcpyHostToDevice);

CudaDST dst(N, CUDA_DST_4);

// Forward transform
dst.execute(d_data);

// Backward transform (DST-4 is self-inverse)
dst.execute(d_data);

// Apply normalization to recover original
double norm = dst.get_normalization();
// Scale d_data by norm on GPU...

cudaMemcpy(h_output, d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
cudaFree(d_data);
// h_output should equal h_input
```

---

## 8. References

1. Ren, M., Gao, Y., Wang, G., & Liu, X. (2020). "Discrete Sine and Cosine Transform and Helmholtz Equation Solver on GPU." *2020 IEEE ISPA/BDCloud/SocialCom/SustainCom*, 57-66. DOI: [10.1109/ISPA-BDCloud-SocialCom-SustainCom51426.2020.00034](https://doi.org/10.1109/ISPA-BDCloud-SocialCom-SustainCom51426.2020.00034)

2. cuHelmholtz GitHub Repository: https://github.com/rmingming/cuHelmholtz

3. FFTW Manual - Real-even/odd DFTs (Cosine/Sine Transforms): http://www.fftw.org/fftw3_doc/Real-even_002fodd-DFTs-_0028Cosine_002fSine-Transforms_0029.html

4. Makhoul, J. (1980). "A Fast Cosine Transform in One and Two Dimensions." *IEEE Trans. Acoust., Speech, Signal Process.*, 28(1), 27-34.

5. Britanak, V., Yip, P., & Rao, K.R. (2007). *Discrete Cosine and Sine Transforms: General Properties, Fast Algorithms and Integer Approximations.* Academic Press.

6. Martucci, S.A. (1994). "Symmetric convolution and the discrete sine and cosine transforms." *IEEE Trans. Signal Process.*, 42(5), 1038-1051.
