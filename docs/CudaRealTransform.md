# CudaRealTransform - CUDA Discrete Cosine and Sine Transform Library

> **Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

CUDA implementation of Discrete Cosine Transform (DCT) and Discrete Sine Transform (DST) Types 1-4, matching FFTW's REDFT/RODFT conventions. This library provides GPU-accelerated transforms using cuFFT as the underlying FFT engine, with efficient algorithms that minimize FFT size while maintaining machine precision accuracy.

The implementation uses unified `CudaRealTransform1D`, `CudaRealTransform2D`, and `CudaRealTransform3D` classes that support both DCT and DST with any combination of transform types across dimensions. Legacy class names (`CudaDCT`, `CudaDST`, etc.) are provided as type aliases for backward compatibility.

## Table of Contents

1. [DCT Types and Definitions](#1-dct-types-and-definitions)
2. [DST Types and Definitions](#2-dst-types-and-definitions)
3. [FFT-based Implementation](#3-fft-based-implementation)
4. [Multi-dimensional Transforms](#4-multi-dimensional-transforms)
5. [Normalization](#5-normalization)
6. [API Reference](#6-api-reference)
7. [Usage Examples](#7-usage-examples)
8. [Algorithm Comparison and Performance](#8-algorithm-comparison-and-performance)
9. [References](#9-references)

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

All DCT types are implemented using FFT with pre/post processing, following the FCT/FST method [1]. This achieves optimal computational efficiency by using the minimum FFT size possible.

| DCT Type | FFT Size | FFT Type | Method |
|----------|----------|----------|--------|
| DCT-1 | N | Z2D | FCT/FST |
| DCT-2 | N | Z2D | FCT/FST |
| DCT-3 | N | D2Z | FCT/FST |
| DCT-4 | 2N | Z2Z | Phase rotation |

### 3.2 DST Algorithm Overview

All DST types are implemented using FFT with pre/post processing, following the FCT/FST method [1]. This achieves optimal computational efficiency by using the minimum FFT size possible.

| DST Type | FFT Size | FFT Type | Method |
|----------|----------|----------|--------|
| DST-1 | N+1 | Z2D | FCT/FST |
| DST-2 | N | Z2D | FCT/FST |
| DST-3 | N | D2Z | FCT/FST |
| DST-4 | 2N | Z2Z | Phase rotation |

The FCT/FST method uses N or N+1 point FFT instead of the traditional 2N or 4N odd extension method, resulting in significant speedup (2-4×).

### 3.3 DCT-1 Algorithm (FCT/FST Method)

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

### 3.5 DST-1 Algorithm (FCT/FST Method)

DST-1 uses an (N+1)-point Z2D (complex-to-real) FFT with the FCT/FST method [1]. The size is N+1 because DST-1 operates on N points but has Dirichlet boundary conditions at both ends (implicit zeros at positions -1 and N).

**Mathematical Basis:**

For an N-point DST-1, we use an (N+1)-point Z2D FFT:
- M = N + 1 (FFT size, where M is the extended size including boundary)
- Input buffer: `[0, x0, x1, ..., x_{N-1}, 0]` (M+1 = N+2 elements)

**Step 1 - Setup:**
Create padded buffer with boundary zeros:
```cpp
buffer[0] = 0            // Left boundary (Dirichlet)
buffer[j] = x[j-1]       // for j = 1..N
buffer[N+1] = 0          // Right boundary (implicit zero, used as padding)
```

**Step 2 - PreOp (Real to Complex):**
Create M/2+1 complex values for Z2D FFT:
```cpp
// Formula: z[k] = (buffer[2k+1] - buffer[2k-1], -buffer[2k])
// with boundary: buffer[M] = 0

for k = 0 to M/2:
    if k == 0:
        z[k] = (2 * buffer[1], 0)              // First: 2*x0
    else if k == M/2:
        z[k] = (-2 * buffer[M-1], 0)           // Last: -2*x_{N-1}
    else:
        z[k].real = buffer[2k+1] - buffer[2k-1]
        z[k].imag = -buffer[2k]
```

**Step 3 - Z2D FFT:**
Execute (N+1)-point complex-to-real inverse FFT:
$$z = \text{IFFT}_{N+1}(\tilde{x})$$

**Step 4 - PostOp (Twiddle Factors with Sine Division):**
Apply sine-based twiddle factors to extract DST-1 coefficients:
```cpp
buffer[0] = 0  // DST-1 first output is always 0
for k = 1 to M/2:
    T_a = (buffer[k] + buffer[M-k]) / (4 * sin(k * π / M))
    T_b = (buffer[k] - buffer[M-k]) / 2
    buffer[k] = (T_a + T_b) / 2
    buffer[M-k] = (T_a - T_b) / 2
```

**Step 5 - Copy:**
Extract result from positions 1..N:
```cpp
Y[k] = buffer[k+1]  for k = 0..N-1
```

**Buffer Layout:**
- Input: N real values `x[0..N-1]`
- Padded buffer: N+2 real values `[0, x0, x1, ..., x_{N-1}, 0]`
- PreOp output: (N+1)/2+1 complex values
- FFT output: N+1 real values
- PostOp output: N+1 real values (position 0 is always 0)
- Final output: N real values `Y[0..N-1]`

**Note:** The key insight is that DST-1 with Dirichlet boundaries at both ends maps to an (N+1)-point problem. The sine division `1/(4*sin(kπ/M))` is the critical twiddle factor that extracts the DST-1 coefficients from the FFT output.

### 3.6 DST-2 Algorithm (FCT/FST Method)

DST-2 uses an N-point Z2D (complex-to-real) FFT with the FCT/FST method [1]. This is the most efficient method, using only N-point FFT instead of 4N-point.

**Mathematical Basis:**

DST-2 can be computed by transforming the input into a specific complex sequence, applying inverse FFT, then extracting the result with twiddle factors:

$$\tilde{x}_k = \begin{cases}
x_0 & k = 0 \\
-x_{N-1} & k = N/2 \\
\frac{x_{2k} - x_{2k-1}}{2} - i\frac{x_{2k} + x_{2k-1}}{2} & 1 \leq k < N/2
\end{cases}$$

**Step 1 - PreOp (Real to Complex):**
```cpp
// Create N/2+1 complex values from N real inputs
for k = 0 to N/2:
    if k == 0:
        out[k] = (x[0], 0)                    // First element
    else if k == N/2:
        out[k] = (-x[N-1], 0)                 // Last element (negated)
    else:
        real = (x[2k] - x[2k-1]) / 2          // Difference
        imag = -(x[2k] + x[2k-1]) / 2         // Negated sum
        out[k] = (real, imag)
```

**Step 2 - Z2D FFT:**
Execute N-point complex-to-real inverse FFT:
$$z = \text{IFFT}_N(\tilde{x})$$

**Step 3 - PostOp (Twiddle Factors):**
Apply sine/cosine twiddle factors to extract DST-2 coefficients:
```cpp
for k = 1 to N:
    if k == N:
        Y[k-1] = 2 * z[0]
    else if k <= N/2:
        sin_k = sin(k * π / (2N))
        cos_k = cos(k * π / (2N))
        T_a = z[k] + z[N-k]
        T_b = z[k] - z[N-k]
        Y[k-1] = T_a * sin_k + T_b * cos_k
    else:
        // Mirror case: k > N/2
        k_mirror = N - k
        sin_k = sin(k_mirror * π / (2N))
        cos_k = cos(k_mirror * π / (2N))
        T_a = z[k_mirror] + z[k]
        T_b = z[k_mirror] - z[k]
        Y[k-1] = T_a * cos_k - T_b * sin_k
```

**Buffer Layout:**
- Input: N real values `x[0..N-1]`
- PreOp output: N/2+1 complex values
- FFT output: N real values
- PostOp output: N real values `Y[0..N-1]`

### 3.7 DST-3 Algorithm (FCT/FST Method)

DST-3 uses an N-point D2Z (real-to-complex) FFT with the FCT/FST method [1]. This is the inverse of DST-2 and uses only N-point FFT instead of 4N-point.

**Mathematical Basis:**

DST-3 is computed by first padding the input with boundary zeros, applying twiddle factors, then forward FFT, and finally extracting the result:

**Step 1 - Setup (Pad with Boundary Zeros):**
Create a padded buffer of size N+2:
```cpp
work[0] = 0              // Left boundary (Dirichlet condition)
work[j] = x[j-1]         // for j = 1..N (data at positions 1 to N)
work[N+1] = 0            // Right boundary (padding for FFT)
```

**Step 2 - PreOp (Twiddle Factors):**
Apply cosine/sine twiddle factors in-place to prepare for FFT:
```cpp
for k = 0 to N/2:
    sin_k = sin(k * π / (2N))
    cos_k = cos(k * π / (2N))
    T_a = work[k] + work[N-k]      // Sum of symmetric pairs
    T_b = work[k] - work[N-k]      // Difference of symmetric pairs
    work[k] = T_a * cos_k + T_b * sin_k
    work[N-k] = T_a * sin_k - T_b * cos_k
```

**Step 3 - D2Z FFT:**
Execute N-point real-to-complex forward FFT:
$$Z = \text{FFT}_N(\text{work}[0..N-1])$$

Output: N/2+1 complex values stored as N+2 interleaved doubles:
```
[Re(Z[0]), Im(Z[0]), Re(Z[1]), Im(Z[1]), ..., Re(Z[N/2]), Im(Z[N/2])]
```

**Step 4 - PostOp (Extract Result):**
Process the complex FFT output to extract DST-3 coefficients:
```cpp
// Negate odd-indexed imaginary parts (positions 3, 5, 7, ...)
for k = 1 to N/2:
    buffer[2k+1] = -buffer[2k+1]

// Extract DST-3 result
Y[0] = 0                           // DST always has Y[0] = 0
Y[1] = Re(Z[0])                    // = buffer[0]
for k = 1 to N/2:
    Y[2k] = buffer[2k+1] - buffer[2k]      // Odd indices
    if 2k+1 < N:
        Y[2k+1] = buffer[2k] + buffer[2k+1]  // Even indices
```

**Step 5 - Copy:**
Copy the result from positions 1..N of the padded buffer to output:
```cpp
out[k] = work[k+1]  for k = 0..N-1
```

**Buffer Layout:**
- Input: N real values `x[0..N-1]`
- Padded buffer: N+2 real values `[0, x0, x1, ..., x_{N-1}, 0]`
- FFT output: N/2+1 complex values = N+2 interleaved doubles
- PostOp output: Result at positions 1..N of buffer
- Final output: N real values `Y[0..N-1]`

**Note:** The FST DST-3 algorithm uses shared memory for the postOp step to efficiently combine the interleaved real/imaginary values. The result matches FFTW's unnormalized RODFT01 output.

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

**DCT-1 (3D):**
- Mirror extension in all three dimensions: [Nx, Ny, Nz] → [2Nx, 2Ny, 2Nz]
- Single 3D D2Z FFT
- Extract coefficients from symmetric positions

**DST-1 (3D):**
- Odd extension in all three dimensions: [Nx, Ny, Nz] → [2(Nx+1), 2(Ny+1), 2(Nz+1)]
- Single 3D D2Z FFT
- Extract coefficients from antisymmetric positions (imaginary parts)

**Note:** Type-1 transforms (DCT-1 and DST-1) use 3D FFT with extension because they cannot be separated into independent 1D transforms when the boundary conditions differ from Types 2-4.

**Types 2,3,4 (3D) - Separable:**
- Batched 1D transforms along Z dimension (preOp → FFT → postOp)
- Batched 1D transforms along Y dimension (preOp → FFT → postOp)
- Batched 1D transforms along X dimension (preOp → FFT → postOp)

### 4.4 3D DST-2 Implementation (FCT/FST Method)

For 3D DST-2, each dimension is transformed sequentially using the FCT/FST method with batched operations:

**Z-direction (innermost, contiguous in memory):**
```
Input layout:  [Nx][Ny][Nz]
Batch size:    Nx * Ny
FFT size:      Nz

Step 1 - PreOp: Convert Nz real values to Nz/2+1 complex values per XY point
        d_data[Nx*Ny*Nz] → d_complex[Nx*Ny*(Nz/2+1)]

Step 2 - Z2D FFT: Batched complex-to-real FFT
        cufftPlanMany(n=Nz, batch=Nx*Ny, inembed=Nz/2+1, onembed=Nz)

Step 3 - PostOp: Apply twiddle factors to extract DST-2 result
        d_work[Nx*Ny*Nz] → d_temp[Nx*Ny*Nz]
```

**Y-direction (requires stride handling):**
```
Input layout:  [Nx][Ny][Nz] stored in d_temp
Batch size:    Nx * Nz
FFT size:      Ny

Step 1 - PreOp: Gather Y-direction data and create complex values
        For each (ix, iz): gather y values at stride Nz

Step 2 - Z2D FFT: Batched FFT with custom strides
        cufftPlanMany(n=Ny, batch=Nx*Nz, inembed=Ny/2+1, onembed=Ny)

Step 3 - PostOp: Apply twiddle factors and scatter back
```

**X-direction (outermost, largest stride):**
```
Input layout:  [Nx][Ny][Nz] stored in d_temp
Batch size:    Ny * Nz
FFT size:      Nx

Step 1 - PreOp: Gather X-direction data and create complex values
        For each (iy, iz): gather x values at stride Ny*Nz

Step 2 - Z2D FFT: Batched FFT with custom strides
        cufftPlanMany(n=Nx, batch=Ny*Nz, inembed=Nx/2+1, onembed=Nx)

Step 3 - PostOp: Apply twiddle factors and write to output
```

### 4.5 3D DST-3 Implementation (FCT/FST Method)

For 3D DST-3, each dimension uses padded buffers for the FCT/FST algorithm:

**Z-direction:**
```
Input layout:  [Nx][Ny][Nz]
Batch size:    Nx * Ny
Buffer size:   Nz + 2 per batch

Step 1 - Setup: Create padded buffer [0, x0, x1, ..., x_{Nz-1}, 0]
        d_data[Nx*Ny*Nz] → d_work[Nx*Ny*(Nz+2)]

Step 2 - PreOp: Apply twiddle factors in-place
        Kernel: 1 block per XY point, Nz/2+1 threads

Step 3 - D2Z FFT: Batched real-to-complex FFT
        cufftPlanMany(n=Nz, batch=Nx*Ny, inembed=Nz+2, onembed=Nz/2+1)

Step 4 - PostOp: Extract DST-3 result using shared memory
        Kernel: 1 block per XY point, shared memory size = (Nz+2)*sizeof(double)

Step 5 - Copy: Extract result from positions 1..Nz
        d_complex → d_temp[Nx*Ny*Nz]
```

**Y-direction:**
```
Input layout:  [Nx][Ny][Nz] stored in d_temp
Batch size:    Nx * Nz
Buffer size:   Ny + 2 per batch

Step 1 - Setup: Create padded buffer with Y-direction gather
        For each (ix, iz): create [0, y0, y1, ..., y_{Ny-1}, 0]

Step 2-4 - Same as Z-direction with Ny dimensions

Step 5 - Copy: Scatter back to Y-direction positions
```

**X-direction:**
```
Input layout:  [Nx][Ny][Nz] stored in d_temp
Batch size:    Ny * Nz
Buffer size:   Nx + 2 per batch

Step 1 - Setup: Create padded buffer with X-direction gather
        For each (iy, iz): create [0, x0, x1, ..., x_{Nx-1}, 0]

Step 2-4 - Same as Z-direction with Nx dimensions

Step 5 - Copy: Scatter back to X-direction positions in output
```

### 4.6 Memory Layout and Buffer Allocation

**Buffer Sizes:**

| Transform Type | Buffer Size per Element | Notes |
|----------------|------------------------|-------|
| DCT-1 | N+1 | Extra point for boundary |
| DCT-2/3 | N | Standard |
| DCT-4 | 2N | Complex buffer for Z2Z |
| DST-1 | N+2 | FCT/FST: padded buffer for Z2D |
| DST-2 | N/2+1 complex | FCT/FST Z2D |
| DST-3 | N+2 | Padded buffer for D2Z |
| DST-4 | 2N | Complex buffer for Z2Z |

**3D Work Buffer Allocation:**
```cpp
// Maximum of all three dimension requirements
size_t work_size_z = Nx * Ny * buffer_size(Nz, type_z);
size_t work_size_y = Nx * Nz * buffer_size(Ny, type_y);
size_t work_size_x = Ny * Nz * buffer_size(Nx, type_x);
size_t max_work = max(work_size_z, work_size_y, work_size_x);
cudaMalloc(&d_work_, max_work);
```

### 4.7 cuFFT Plan Configuration for DST-2/3

**DST-2 (Z2D FFT):**
```cpp
// Z-direction example
int n[1] = {Nz};                    // FFT size
int inembed[1] = {Nz / 2 + 1};      // Input: complex array
int onembed[1] = {Nz};              // Output: real array
cufftPlanMany(&plan, 1, n,
    inembed, 1, Nz / 2 + 1,         // Input stride, distance
    onembed, 1, Nz,                  // Output stride, distance
    CUFFT_Z2D, Nx * Ny);            // Type, batch count
```

**DST-3 (D2Z FFT with padded buffer):**
```cpp
// Z-direction example
int n[1] = {Nz};                    // FFT size
int inembed[1] = {Nz + 2};          // Input: padded real array
int onembed[1] = {Nz / 2 + 1};      // Output: complex array
cufftPlanMany(&plan, 1, n,
    inembed, 1, Nz + 2,             // Input stride, distance (padded)
    onembed, 1, Nz / 2 + 1,         // Output stride, distance
    CUFFT_D2Z, Nx * Ny);            // Type, batch count
```

**Key Difference:** DST-3 uses input stride `Nz + 2` to account for the padded buffer layout `[0, x0, ..., x_{Nz-1}, 0]`, where the FFT operates on positions 0 to Nz-1.

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

### 6.1 Transform Type Enumeration

```cpp
enum CudaTransformType {
    // DCT Types (Neumann boundary conditions)
    CUDA_DCT_1 = 0,   // FFTW_REDFT00: DCT-I
    CUDA_DCT_2 = 1,   // FFTW_REDFT10: DCT-II
    CUDA_DCT_3 = 2,   // FFTW_REDFT01: DCT-III
    CUDA_DCT_4 = 3,   // FFTW_REDFT11: DCT-IV
    // DST Types (Dirichlet boundary conditions)
    CUDA_DST_1 = 4,   // FFTW_RODFT00: DST-I
    CUDA_DST_2 = 5,   // FFTW_RODFT10: DST-II
    CUDA_DST_3 = 6,   // FFTW_RODFT01: DST-III
    CUDA_DST_4 = 7    // FFTW_RODFT11: DST-IV
};

// Helper functions
bool isDCT(CudaTransformType type);  // Returns true for DCT types
bool isDST(CudaTransformType type);  // Returns true for DST types
bool isType1(CudaTransformType type); // Returns true for Type-1 transforms
const char* getTransformName(CudaTransformType type); // Returns "DCT-1", "DST-2", etc.
```

### 6.2 CudaRealTransform1D

```cpp
class CudaRealTransform1D {
public:
    // Constructor: N = transform size, type = transform type (DCT/DST 1-4)
    CudaRealTransform1D(int N, CudaTransformType type);
    ~CudaRealTransform1D();

    // Execute transform in-place on device array
    void execute(double* d_data);

    // Get input/output size (N for types 2,3,4; N+1 for DCT-1; N-1 for DST-1)
    int get_size() const;

    // Get transform type
    CudaTransformType get_type() const;

    // Get normalization factor for round-trip
    double get_normalization() const;
};
```

### 6.3 CudaRealTransform2D

```cpp
class CudaRealTransform2D {
public:
    // Constructor: same type for both dimensions
    CudaRealTransform2D(int Nx, int Ny, CudaTransformType type);

    // Constructor: different types per dimension
    // Note: Type-1 transforms cannot be mixed with other types
    CudaRealTransform2D(int Nx, int Ny, CudaTransformType type_x, CudaTransformType type_y);
    ~CudaRealTransform2D();

    void execute(double* d_data);
    int get_size() const;
    void get_dims(int& nx, int& ny) const;
    void get_types(CudaTransformType& type_x, CudaTransformType& type_y) const;
    double get_normalization() const;
};
```

### 6.4 CudaRealTransform3D

```cpp
class CudaRealTransform3D {
public:
    // Constructor: same type for all dimensions
    CudaRealTransform3D(int Nx, int Ny, int Nz, CudaTransformType type);

    // Constructor: different types per dimension
    // Note: Type-1 transforms cannot be mixed with other types
    CudaRealTransform3D(int Nx, int Ny, int Nz,
                        CudaTransformType type_x, CudaTransformType type_y, CudaTransformType type_z);
    ~CudaRealTransform3D();

    void execute(double* d_data);
    int get_size() const;
    void get_dims(int& nx, int& ny, int& nz) const;
    void get_types(CudaTransformType& type_x, CudaTransformType& type_y, CudaTransformType& type_z) const;
    double get_normalization() const;
};
```

### 6.5 Backward Compatibility Aliases

For backward compatibility, the following type aliases are provided:

```cpp
// Type aliases for old class names
typedef CudaRealTransform1D CudaDCT;
typedef CudaRealTransform1D CudaDST;
typedef CudaRealTransform2D CudaDCT2D;
typedef CudaRealTransform2D CudaDST2D;
typedef CudaRealTransform2D CudaMixedTransform2D;
typedef CudaRealTransform3D CudaDCT3D;
typedef CudaRealTransform3D CudaDST3D;
typedef CudaRealTransform3D CudaMixedTransform3D;

// Type aliases for old enum names
typedef CudaTransformType CudaDCTType;
typedef CudaTransformType CudaDSTType;
```

---

## 7. Usage Examples

> **Note:** These examples use the new unified class names (`CudaRealTransform1D/2D/3D`).
> Legacy class names (`CudaDCT`, `CudaDST`, etc.) are still supported for backward compatibility.

### 7.1 1D DCT-4

```cpp
#include "CudaRealTransform.h"

int N = 64;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * N);
cudaMemcpy(d_data, h_input, sizeof(double) * N, cudaMemcpyHostToDevice);

CudaRealTransform1D dct(N, CUDA_DCT_4);
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

CudaRealTransform1D dst(N, CUDA_DST_4);
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

CudaRealTransform2D dst(Nx, Ny, CUDA_DST_1);
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

CudaRealTransform3D dst(Nx, Ny, Nz, CUDA_DST_4);
dst.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 7.5 2D Mixed Transform (DST-2 in X, DST-3 in Y)

```cpp
int Nx = 32, Ny = 64;
int M = Nx * Ny;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * M);
cudaMemcpy(d_data, h_input, sizeof(double) * M, cudaMemcpyHostToDevice);

// Different transform types per dimension
CudaRealTransform2D transform(Nx, Ny, CUDA_DST_2, CUDA_DST_3);
transform.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 7.6 3D Mixed Transform (DCT in X/Y, DST in Z)

```cpp
int Nx = 32, Ny = 32, Nz = 32;
int M = Nx * Ny * Nz;

double* d_data;
cudaMalloc(&d_data, sizeof(double) * M);
cudaMemcpy(d_data, h_input, sizeof(double) * M, cudaMemcpyHostToDevice);

// DCT-2 in X, DCT-2 in Y, DST-2 in Z (Neumann in XY, Dirichlet in Z)
CudaRealTransform3D transform(Nx, Ny, Nz, CUDA_DCT_2, CUDA_DCT_2, CUDA_DST_2);
transform.execute(d_data);

cudaMemcpy(h_output, d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

### 7.7 Round-trip with Normalization

```cpp
int N = 64;
double* d_data;
cudaMalloc(&d_data, sizeof(double) * N);
cudaMemcpy(d_data, h_input, sizeof(double) * N, cudaMemcpyHostToDevice);

CudaRealTransform1D dst(N, CUDA_DST_4);

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

## 8. Algorithm Comparison and Performance

### 8.1 FCT/FST vs Traditional Odd Extension

The traditional approach for DST uses odd-symmetric extension which requires larger FFT sizes. The FCT/FST method [1] minimizes FFT size:

| Transform | Traditional FFT Size | FCT/FST FFT Size | Speedup |
|-----------|---------------------|---------------------|---------|
| DST-1 | 2(N+1) | N+1 | ~2× |
| DST-2 | 4N | N | ~4× |
| DST-3 | 4N | N | ~4× |
| DST-4 | 2N (phase rotation) | 2N | - |

**Detailed Comparison:**

| Method | FFT Size | Memory | Complexity |
|--------|----------|--------|------------|
| Odd Extension (DST-1) | 2(N+1) | O(2N) | O(2N log 2N) |
| FCT/FST (DST-1) | N+1 | O(N+2) | O(N log N) |
| Odd Extension (DST-2) | 4N | O(4N) | O(4N log 4N) |
| FCT/FST (DST-2) | N | O(N) | O(N log N) |
| Odd Extension (DST-3) | 4N | O(4N) | O(4N log 4N) |
| FCT/FST (DST-3) | N | O(N+2) | O(N log N) |

**Performance Improvement:**
- DST-1: FFT size reduced by 2×, ~2× speedup
- DST-2/3: FFT size reduced by 4×, ~4× speedup
- Memory usage reduced proportionally

### 8.2 Algorithm Summary Table

| Transform | FFT Type | FFT Size | Buffer Size | PreOp | PostOp |
|-----------|----------|----------|-------------|-------|--------|
| DCT-1 | Z2D | N | N+2 | Reduction + prepare | Sine division |
| DCT-2 | Z2D | N | N/2+1 complex | Reorder + half | Twiddle multiply |
| DCT-3 | D2Z | N | N | Scale + prepare | Extract real |
| DCT-4 | Z2Z | 2N | 2N complex | Phase multiply | Phase extract |
| DST-1 | Z2D | N+1 | N+2 | Diff pairs | Sine division |
| DST-2 | Z2D | N | N/2+1 complex | Diff/sum pairs | Twiddle multiply |
| DST-3 | D2Z | N | N+2 | Twiddle factors | Sum/diff extract |
| DST-4 | Z2Z | 2N | 2N complex | Phase multiply | Phase extract (imag) |

### 8.3 CUDA Kernel Design

**DST-2 Kernel Structure (FCT/FST):**
```
PreOp Kernel:
  - 1 thread per complex output element
  - Grid: (Nx*Ny*(Nz/2+1) + 255) / 256 blocks
  - No shared memory required

PostOp Kernel:
  - 1 thread per output element
  - Grid: (Nx*Ny*Nz + 255) / 256 blocks
  - Uses sincos() for twiddle factors
```

**DST-3 Kernel Structure (FCT/FST):**
```
Setup Kernel:
  - 1 thread per padded buffer element
  - Grid: (Nx*Ny*(Nz+2) + 255) / 256 blocks

PreOp Kernel:
  - 1 block per batch (XY point for Z-direction)
  - N/2+1 threads per block
  - Uses sincos() for twiddle factors

PostOp Kernel:
  - 1 block per batch
  - N/2+1 threads per block
  - Shared memory: (N+2) * sizeof(double)
  - Complex output reinterpretation

Copy Kernel:
  - 1 thread per output element
  - Extract from positions 1..N of padded buffer
```

### 8.4 Numerical Accuracy

All implementations achieve machine precision accuracy when compared to FFTW:

| Transform | Max Relative Error (1D) | Max Relative Error (3D) |
|-----------|------------------------|------------------------|
| DST-1 | < 3×10⁻¹⁶ | < 3×10⁻¹⁶ |
| DST-2 | < 4×10⁻¹⁶ | < 3×10⁻¹⁶ |
| DST-3 | < 2×10⁻¹⁶ | < 4×10⁻¹⁶ |
| DST-4 | < 3×10⁻¹⁶ | < 4×10⁻¹⁶ |

---

## 9. References

1. Makhoul, J. (1980). "A Fast Cosine Transform in One and Two Dimensions." *IEEE Trans. Acoust., Speech, Signal Process.*, 28(1), 27-34. DOI: [10.1109/TASSP.1980.1163351](https://doi.org/10.1109/TASSP.1980.1163351)
   - **Note:** This seminal paper introduced the **Fast Cosine Transform (FCT)** terminology and the N-point FFT formulation with twiddle factors, reducing computation by half compared to the traditional 2N-point DFT approach.

2. Martucci, S.A. (1994). "Symmetric convolution and the discrete sine and cosine transforms." *IEEE Trans. Signal Process.*, 42(5), 1038-1051.
   - Extends Makhoul's approach to DST, establishing the **Fast Sine Transform (FST)** algorithms.

3. Britanak, V., Yip, P., & Rao, K.R. (2007). *Discrete Cosine and Sine Transforms: General Properties, Fast Algorithms and Integer Approximations.* Academic Press.
   - Comprehensive reference for DCT/DST theory and algorithms.

4. Ren, M., Gao, Y., Wang, G., & Liu, X. (2020). "Discrete Sine and Cosine Transform and Helmholtz Equation Solver on GPU." *2020 IEEE ISPA/BDCloud/SocialCom/SustainCom*, 57-66. DOI: [10.1109/ISPA-BDCloud-SocialCom-SustainCom51426.2020.00034](https://doi.org/10.1109/ISPA-BDCloud-SocialCom-SustainCom51426.2020.00034)
   - CUDA implementation of FCT/FST algorithms (cuHelmholtz library).

5. cuHelmholtz GitHub Repository: https://github.com/rmingming/cuHelmholtz

6. FFTW Manual - Real-even/odd DFTs (Cosine/Sine Transforms): http://www.fftw.org/fftw3_doc/Real-even_002fodd-DFTs-_0028Cosine_002fSine-Transforms_0029.html
