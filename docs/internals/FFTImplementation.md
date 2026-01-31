# FFT Implementation

> **Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the FFT implementations used in the polymer field theory library, including DCT/DST for non-periodic boundaries and crystallographic FFT (CrysFFT) for space-group symmetry acceleration.

## Table of Contents

1. [Overview](#1-overview)
2. [Discrete Cosine Transform (DCT)](#2-discrete-cosine-transform-dct)
3. [Discrete Sine Transform (DST)](#3-discrete-sine-transform-dst)
4. [Multi-dimensional Transforms](#4-multi-dimensional-transforms)
5. [Normalization](#5-normalization)
6. [Crystallographic FFT (CrysFFT)](#6-crystallographic-fft-crysfft)
7. [Implementation Notes](#7-implementation-notes)
8. [Performance Benchmarks](#8-performance-benchmarks)
9. [References](#9-references)

---

## 1. Overview

The library provides spectral transforms for solving the modified diffusion equation:

| Transform | Boundary Condition | Physical Meaning |
|-----------|-------------------|------------------|
| FFT | Periodic | Cyclic boundary |
| DCT | Reflecting | Neumann (zero flux) |
| DST | Absorbing | Dirichlet (zero value) |

**Implementations:**
- **CPU**: FFTW library (REDFT/RODFT for DCT/DST)
- **CUDA**: cuFFT for FFT, custom `CudaRealTransform` for DCT/DST

---

## 2. Discrete Cosine Transform (DCT)

### 2.1 Types and Definitions

| Type | FFTW Name | Formula | Grid Points |
|------|-----------|---------|-------------|
| DCT-1 | REDFT00 | $X_k = x_0 + (-1)^k x_{N} + 2\sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi k n}{N}\right)$ | N+1 |
| DCT-2 | REDFT10 | $X_k = 2\sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k (2n+1)}{2N}\right)$ | N |
| DCT-3 | REDFT01 | $X_k = x_0 + 2\sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi n (2k+1)}{2N}\right)$ | N |
| DCT-4 | REDFT11 | $X_k = 2\sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi (2n+1)(2k+1)}{4N}\right)$ | N |

### 2.2 Boundary Conditions and Symmetry

| Type | Left Boundary | Right Boundary | Physical Meaning |
|------|---------------|----------------|------------------|
| DCT-1 | Even at point | Even at point | Neumann-Neumann |
| DCT-2 | Even at half-point | Even at point | Mixed boundary |
| DCT-3 | Even at point | Even at half-point | Mixed boundary |
| DCT-4 | Even at half-point | Even at half-point | Half-sample symmetric |

**Inverse relationships:**
- DCT-1 is self-inverse: DCT-1(DCT-1(x)) = 2N · x
- DCT-2 and DCT-3 are inverses: DCT-3(DCT-2(x)) = 2N · x
- DCT-4 is self-inverse: DCT-4(DCT-4(x)) = 2N · x

### 2.3 FCT Algorithm (CUDA)

The CUDA implementation uses the Fast Cosine Transform (FCT) method [1], which uses N-point FFT instead of the traditional 2N-point odd extension:

| DCT Type | FFT Size | FFT Type | Speedup |
|----------|----------|----------|---------|
| DCT-1 | N | Z2D | ~2x |
| DCT-2 | N | Z2D | ~2x |
| DCT-3 | N | D2Z | ~2x |
| DCT-4 | 2N | Z2Z | Phase rotation |

---

## 3. Discrete Sine Transform (DST)

### 3.1 Types and Definitions

| Type | FFTW Name | Formula | Grid Points |
|------|-----------|---------|-------------|
| DST-1 | RODFT00 | $Y_k = 2\sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (n+1)(k+1)}{N+1}\right)$ | N |
| DST-2 | RODFT10 | $Y_k = 2\sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (2n+1)(k+1)}{2N}\right)$ | N |
| DST-3 | RODFT01 | $Y_k = (-1)^k x_{N-1} + 2\sum_{n=0}^{N-2} x_n \sin\left(\frac{\pi (n+1)(2k+1)}{2N}\right)$ | N |
| DST-4 | RODFT11 | $Y_k = 2\sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (2n+1)(2k+1)}{4N}\right)$ | N |

### 3.2 Boundary Conditions and Symmetry

| Type | Left Boundary | Right Boundary | Physical Meaning |
|------|---------------|----------------|------------------|
| DST-1 | Odd at point | Odd at point | Dirichlet-Dirichlet |
| DST-2 | Odd at half-point | Odd at point | Mixed boundary |
| DST-3 | Odd at point | Odd at half-point | Mixed boundary |
| DST-4 | Odd at half-point | Odd at half-point | Half-sample antisymmetric |

**Inverse relationships:**
- DST-1 is self-inverse: DST-1(DST-1(x)) = 2(N+1) · x
- DST-2 and DST-3 are inverses: DST-3(DST-2(x)) = 2N · x
- DST-4 is self-inverse: DST-4(DST-4(x)) = 2N · x

### 3.3 FST Algorithm (CUDA)

| DST Type | FFT Size | FFT Type | Traditional Size | Speedup |
|----------|----------|----------|-----------------|---------|
| DST-1 | N+1 | Z2D | 2(N+1) | ~2x |
| DST-2 | N | Z2D | 4N | ~4x |
| DST-3 | N | D2Z | 4N | ~4x |
| DST-4 | 2N | Z2Z | 2N | Phase rotation |

---

## 4. Multi-dimensional Transforms

### Supported Configurations

| Dimension | Supported Types | Mixed Types |
|-----------|-----------------|-------------|
| 1D | DCT/DST-1,2,3,4 | N/A |
| 2D | DCT/DST-1,2,3,4 | DCT/DST-2,3,4 |
| 3D | DCT/DST-1,2,3,4 | DCT/DST-2,3,4 |

**Note**: Type-1 transforms cannot be mixed with other types due to different point counts.

### 3D Transform Algorithm

**Types 2,3,4 (Separable):**
- Batched 1D transforms along Z dimension (innermost)
- Batched 1D transforms along Y dimension
- Batched 1D transforms along X dimension (outermost)

**Type-1 (Non-separable):**
- Mirror/odd extension in all three dimensions
- Single 3D D2Z FFT
- Extract coefficients from symmetric/antisymmetric positions

### Memory Layout

| Transform Type | Buffer Size per Element |
|----------------|------------------------|
| DCT-1 | N+1 |
| DCT-2/3 | N |
| DCT-4 | 2N (complex) |
| DST-1 | N+2 (FCT/FST padded) |
| DST-2 | N/2+1 complex |
| DST-3 | N+2 (padded) |
| DST-4 | 2N (complex) |

---

## 5. Normalization

Both DCT and DST follow FFTW's unnormalized convention.

### DCT Normalization Factors

| Transform | Round-trip Scale |
|-----------|------------------|
| 1D DCT-1/2/3/4 | 2N |
| 2D (all types) | 4 N_x N_y |
| 3D (all types) | 8 N_x N_y N_z |

### DST Normalization Factors

| Transform | Round-trip Scale |
|-----------|------------------|
| 1D DST-1 | 2(N+1) |
| 1D DST-2/3/4 | 2N |
| 2D DST-1 | 4(N_x+1)(N_y+1) |
| 2D DST-2/3/4 | 4 N_x N_y |
| 3D DST-1 | 8(N_x+1)(N_y+1)(N_z+1) |
| 3D DST-2/3/4 | 8 N_x N_y N_z |

---

## 6. Crystallographic FFT (CrysFFT)

CrysFFT replaces the standard 3D FFT with symmetry-reduced transforms when a compatible space group is enabled.

### 6.1 Terminology

- **Logical grid**: Full simulation grid, size `Nx x Ny x Nz`
- **Physical grid**: CrysFFT working grid (reduced by symmetry)
- **Irreducible grid**: Smallest symmetry-unique grid (one point per orbit)

### 6.2 Algorithms

#### Pmmm DCT (Mirror Symmetry)

For space groups with three perpendicular mirror planes:

```
q_out = DCT-III[ exp(-k^2 * coeff) * DCT-II[q_in] ]
```

- Physical grid: `(Nx/2) x (Ny/2) x (Nz/2)`
- Wavenumbers: k = π * i / L

#### 3m Recursive (2x2x2) Algorithm

For space groups with 3m translations (mirror + translational offsets):

- Same physical grid size `(Nx/2) x (Ny/2) x (Nz/2)`
- Uses eight sub-transforms with precomputed twiddle factors
- Typically faster than Pmmm DCT when available

#### ObliqueZ (z-mirror)

For space groups with z-mirror and α=β=90° (γ arbitrary):

```
q_out = DCT-III_z[ exp(-k^2 * coeff) * FFT_xy[ DCT-II_z[q_in] ] ]
```

- Physical grid: `Nx x Ny x (Nz/2)`
- Supports oblique in-plane angles (e.g., hexagonal γ=120°)
- For glide mirror (t_z=1/2): requires `Nz % 4 == 0`

### 6.3 Activation Criteria

CrysFFT is enabled automatically when:
- `space_group` is set
- Dimension is 3D
- Boundary conditions are periodic
- `Nx`, `Ny`, `Nz` are even
- Fields are real-valued

**Selection order:**
1. **3m recursive** if space group provides 3m translations and `Nz/2 % 8 == 0`
2. **Pmmm DCT** if space group has mirror planes in x/y/z
3. **ObliqueZ** if space group has z-mirror with α=β=90°
4. Fall back to standard FFT

### 6.4 Physical Basis Choices

| Basis | Grid Size | Mapping Overhead |
|-------|-----------|-----------------|
| Irreducible | Minimal (one per orbit) | Gather/scatter required |
| Pmmm physical | (Nx/2)(Ny/2)(Nz/2) | Identity (none) |
| 3m physical | (Nx/2)(Ny/2)(Nz/2) | Identity (none) |
| Z-mirror physical | Nx * Ny * (Nz/2) | Identity (none) |

The solver automatically selects the fastest compatible physical basis.

### 6.5 Phase Support Table

| Phase | Space group | 3m recursive | Pmmm DCT | ObliqueZ |
|-------|-------------|:------------:|:--------:|:--------:|
| BCC | Im-3m (529) | ✓ | ✓ | |
| FCC | Fm-3m (523) | ✓ | ✓ | |
| SC | Pm-3m (517) | ✓ | ✓ | |
| A15 | Pm-3n (520) | ✓ | ✓ | |
| DG (Gyroid) | Ia-3d (530) | ✓ | ✗ | |
| DD | Pn-3m (522) | ✓ | ✗ | |
| SG | I4_132 (510) | ✗ | ✗ | |
| HCP | P6_3/mmc (488) | ✗ | ✗ | ✓ |
| PL | P6/mmm (485) | ✗ | ✗ | ✓ |

---

## 7. Implementation Notes

### 7.1 CPU

- `FftwFFT<T, DIM>`: Standard FFT using FFTW3
- `FftwCrysFFTPmmm`: Pmmm DCT using FFTW REDFT10/REDFT01
- `FftwCrysFFTRecursive3m`: Recursive 3m algorithm
- `FftwCrysFFTObliqueZ`: DCT-z + FFT-xy

**Threading**: FFTW multithreading is disabled; concurrency is handled at the solver level (OpenMP over propagators). Thread-local buffers are used to avoid race conditions.

### 7.2 CUDA

- `CudaFFT<T, DIM>`: Standard FFT using cuFFT
- `CudaRealTransform1D/2D/3D`: DCT/DST using FCT/FST algorithms
- `CudaCrysFFT`: Pmmm DCT via CudaRealTransform
- `CudaCrysFFTRecursive3m`: Recursive 3m on GPU with stream-aware diffusion
- `CudaCrysFFTObliqueZ`: DCT-z + FFT-xy

**Backward Compatibility:**

```cpp
// Legacy class names (still supported)
typedef CudaRealTransform1D CudaDCT;
typedef CudaRealTransform1D CudaDST;
typedef CudaRealTransform2D CudaDCT2D;
typedef CudaRealTransform3D CudaDCT3D;
```

### 7.3 API Reference

```cpp
// Transform type enumeration
enum CudaTransformType {
    CUDA_DCT_1 = 0, CUDA_DCT_2 = 1, CUDA_DCT_3 = 2, CUDA_DCT_4 = 3,
    CUDA_DST_1 = 4, CUDA_DST_2 = 5, CUDA_DST_3 = 6, CUDA_DST_4 = 7
};

// 3D transform
class CudaRealTransform3D {
public:
    // Same type for all dimensions
    CudaRealTransform3D(int Nx, int Ny, int Nz, CudaTransformType type);

    // Different types per dimension
    CudaRealTransform3D(int Nx, int Ny, int Nz,
                        CudaTransformType type_x,
                        CudaTransformType type_y,
                        CudaTransformType type_z);

    void execute(double* d_data);
    double get_normalization() const;
};
```

**Usage Example:**

```cpp
int Nx = 32, Ny = 32, Nz = 32;
double* d_data;
cudaMalloc(&d_data, sizeof(double) * Nx * Ny * Nz);

// DCT-2 in X/Y, DST-2 in Z (Neumann in XY, Dirichlet in Z)
CudaRealTransform3D transform(Nx, Ny, Nz, CUDA_DCT_2, CUDA_DCT_2, CUDA_DST_2);
transform.execute(d_data);
```

---

## 8. Performance Benchmarks

### 8.1 DCT/DST Numerical Accuracy

All DCT/DST implementations achieve machine precision (< 4×10⁻¹⁶ relative error) when compared to FFTW.

### 8.2 CrysFFT Benchmark (64³, Im-3m BCC)

**Setup**: nx=[64,64,64], coeff=0.01, 200 iterations, NVIDIA A10 GPU, CPU (4 threads)

| Platform | Standard FFT | 3m physical | Pmmm physical |
|----------|-------------|-------------|---------------|
| CUDA | 0.177 ms | 0.056 ms (3.1x) | 0.091 ms (2.0x) |
| CPU | 2.09 ms | 0.41 ms (5.1x) | 0.58 ms (3.6x) |

### 8.3 ObliqueZ Benchmark (64³, Hexagonal P6/mmm)

**Setup**: nx=[64,64,64], γ=120°, coeff=0.01, 200 iterations, NVIDIA A10 GPU, CPU (4 threads)

| Platform | Standard FFT | ObliqueZ |
|----------|-------------|----------|
| CUDA | 0.132 ms | 0.120 ms (1.1x) |
| CPU | 2.00 ms | 1.40 ms (1.4x) |

---

## 9. References

1. Makhoul, J. "A Fast Cosine Transform in One and Two Dimensions." *IEEE Trans. Acoust., Speech, Signal Process.*, **28**(1), 27-34 (1980).

2. Martucci, S.A. "Symmetric convolution and the discrete sine and cosine transforms." *IEEE Trans. Signal Process.*, **42**(5), 1038-1051 (1994).

3. Britanak, V., Yip, P., & Rao, K.R. *Discrete Cosine and Sine Transforms: General Properties, Fast Algorithms and Integer Approximations.* Academic Press (2007).

4. Qiang, Y. & Li, W. "Accelerated pseudo-spectral method of self-consistent field theory via crystallographic fast Fourier transform." *Macromolecules* **53**, 9943-9952 (2020).

5. Ten Eyck, L. F. "Crystallographic Fast Fourier Transforms." *Acta Cryst.* **A29**, 183-191 (1973).

6. FFTW Manual - Real-even/odd DFTs: http://www.fftw.org/fftw3_doc/
