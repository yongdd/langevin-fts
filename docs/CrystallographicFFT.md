# Crystallographic FFT using Discrete Cosine Transform

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the DCT-based Crystallographic Fast Fourier Transform (CFFT) algorithm for accelerating pseudo-spectral computations in polymer field theory simulations.

**Status**: Planned feature (not yet implemented)

## Table of Contents

1. [Overview](#1-overview)
2. [Theoretical Background](#2-theoretical-background)
3. [Algorithm](#3-algorithm)
4. [Applicable Space Groups](#4-applicable-space-groups)
5. [Complexity Analysis](#5-complexity-analysis)
6. [Implementation Notes](#6-implementation-notes)
7. [References](#7-references)

---

## 1. Overview

**CFFT** (Crystallographic Fast Fourier Transform) exploits crystallographic symmetry to accelerate FFT computations by performing calculations only on the asymmetric unit using Discrete Cosine Transforms.

**Key idea**: For centrosymmetric space groups with 3 perpendicular mirror/glide planes, the 3D FFT on a full grid can be replaced by a 3D DCT on a grid reduced by a factor of 8 or more.

**Benefits:**
- ~8x speedup from mirror symmetry (1/8 grid size)
- Additional 2-4x from lattice centering (I: 2x, F: 4x)
- Real arithmetic only (DCT produces real output)
- Memory reduction proportional to grid reduction

---

## 2. Theoretical Background

### 2.1 Mirror Symmetry and DCT

For a function with mirror symmetry about $x = 0$:

$$f(x) = f(-x)$$

The Fourier transform becomes purely real (cosine series only):

$$F(k) = \int f(x) e^{-ikx} dx = 2 \int_0^{\infty} f(x) \cos(kx) dx$$

This is exactly the Discrete Cosine Transform (DCT). Therefore:
- **FFT** (2N points, complex) → **DCT** (N points, real)
- Computational savings: ~4x per dimension

### 2.2 Three Perpendicular Mirrors

For space groups with mirrors perpendicular to all three axes:

$$f(x, y, z) = f(-x, y, z) = f(x, -y, z) = f(x, y, -z)$$

The 3D FFT reduces to a 3D DCT on 1/8 of the grid:

| Grid | Size |
|------|------|
| Full grid | $(N_x, N_y, N_z)$ |
| Reduced grid | $(N_x/2, N_y/2, N_z/2)$ |

### 2.3 Centrosymmetry Requirement

DCT produces real output, which requires the Fourier transform to have Hermitian symmetry:

$$F(\mathbf{k}) = F^*(-\mathbf{k})$$

This holds for **centrosymmetric space groups** (those with inversion centers), which includes all point groups: mmm, 4/mmm, 6/mmm, m-3m, m-3.

### 2.4 Lattice Centering

Additional reduction comes from lattice centering:

| Centering | Description | Reduction Factor |
|-----------|-------------|------------------|
| P (Primitive) | No centering | 1x |
| I (Body-centered) | Center at $(1/2, 1/2, 1/2)$ | 2x |
| F (Face-centered) | Centers on all faces | 4x |

**Total reduction: 8 × centering factor**

---

## 3. Algorithm

### 3.1 Forward Transform (Real Space → Reciprocal Space)

```
Input:  f(x,y,z) on reduced grid [0, Nx/2) × [0, Ny/2) × [0, Nz/2)
Output: F(kx,ky,kz) on reduced grid

F = DCT-II(f)    # 3D Type-II DCT
```

### 3.2 Inverse Transform (Reciprocal Space → Real Space)

```
Input:  F(kx,ky,kz) on reduced grid
Output: f(x,y,z) on reduced grid

f = IDCT-II(F)   # 3D Inverse Type-II DCT (= Type-III DCT)
```

### 3.3 SCFT Diffusion Step

The modified diffusion equation:

$$\frac{\partial q}{\partial s} = \nabla^2 q - w \cdot q$$

Pseudo-spectral solution (one step):

$$q(s+\Delta s) \approx e^{-\frac{\Delta s}{2} w} \cdot \text{IDCT}\left[ e^{-\Delta s \cdot k^2} \cdot \text{DCT}\left[ e^{-\frac{\Delta s}{2} w} \cdot q(s) \right] \right]$$

**Implementation:**

```python
def diffusion_step(q, w, k_squared, delta_s):
    q1 = exp(-delta_s/2 * w) * q          # Real space
    Q1 = DCT(q1)                           # → Reciprocal space
    Q2 = exp(-delta_s * k_squared) * Q1    # Reciprocal space
    q2 = IDCT(Q2)                          # → Real space
    return exp(-delta_s/2 * w) * q2        # Real space
```

All operations are on the reduced grid (1/8 size or smaller).

### 3.4 Expansion to Full Grid

When full grid output is needed (e.g., for visualization):

```python
def expand_to_full(f_reduced, nx, ny, nz):
    # f_reduced has shape (nx/2, ny/2, nz/2)
    # f_full will have shape (nx, ny, nz)

    f_full = zeros(nx, ny, nz)
    rx, ry, rz = nx//2, ny//2, nz//2

    # Copy asymmetric unit
    f_full[:rx, :ry, :rz] = f_reduced

    # Mirror in x
    f_full[rx:, :ry, :rz] = flip(f_reduced, axis=0)

    # Mirror in y
    f_full[:, ry:, :rz] = flip(f_full[:, :ry, :rz], axis=1)

    # Mirror in z
    f_full[:, :, rz:] = flip(f_full[:, :, :rz], axis=2)

    return f_full
```

---

## 4. Applicable Space Groups

### 4.1 Requirements

1. **Centrosymmetric** (has inversion center)
2. **Has 3 mutually perpendicular mirror or glide planes**

### 4.2 Coverage

**69 out of 230 space groups (30%)** satisfy these requirements.

| Point Group | Crystal System | Space Groups | Count |
|-------------|----------------|--------------|-------|
| mmm | Orthorhombic | #47-74 | 28 |
| 4/mmm | Tetragonal | #123-142 | 20 |
| 6/mmm | Hexagonal | #191-194 | 4 |
| m-3m | Cubic | #221-230 | 10 |
| m-3 | Cubic | #200-206 | 7 |

### 4.3 Common Block Copolymer Phases

| Phase | Space Group | Number | Centering | Total Reduction |
|-------|-------------|--------|-----------|-----------------|
| BCC | Im-3m | 229 | I | 16x |
| FCC | Fm-3m | 225 | F | 32x |
| Gyroid | Ia-3d | 230 | I | 16x |
| Diamond | Fd-3m | 227 | F | 32x |
| HCP | P6_3/mmc | 194 | P | 8x |
| PL | P6_3/mmc | 194 | P | 8x |
| A15 | Pm-3n | 223 | P | 8x |
| Sigma (σ) | P4_2/mnm | 136 | P | 8x |
| C14 | P6_3/mmc | 194 | P | 8x |
| C15 | Fd-3m | 227 | F | 32x |
| O70 | Fddd | 70 | F | 32x |
| Z | P6/mmm | 191 | P | 8x |

---

## 5. Complexity Analysis

### 5.1 Computational Cost

| Operation | Standard FFT | CFFT |
|-----------|--------------|------|
| Grid size | $N^3$ | $(N/2)^3 = N^3/8$ |
| Transform | $O(N^3 \log N^3)$ | $O((N/2)^3 \log (N/2)^3)$ |
| Memory | $O(N^3)$ complex | $O(N^3/8)$ real |
| **Speedup** | 1x | **~8x** |

### 5.2 Expected Performance

Benchmarks on $128^3$ grid (vs SciPy FFT):

| Phase | Space Group | Reduction | Speedup |
|-------|-------------|-----------|---------|
| BCC | Im-3m | 16x | ~18x |
| FCC | Fm-3m | 32x | ~17x |
| Gyroid | Ia-3d | 16x | ~18x |
| Sigma | P4_2/mnm | 8x | ~20x |

**Note:** vs FFTW, expect ~60% of these speedups (FFTW is faster than SciPy).

---

## 6. Implementation Notes

### 6.1 DCT Type Selection

Use **DCT-II** (forward) and **DCT-III** (inverse), which are the standard "DCT" and "IDCT" in most libraries.

With orthonormal normalization (`norm='ortho'` in SciPy):
- DCT-II and DCT-III are exact inverses
- No additional normalization factors needed

### 6.2 Wavevector Calculation

For DCT on a reduced grid with box length $L$:

$$k_i = \frac{\pi i}{L}, \quad i = 0, 1, \ldots, N/2 - 1$$

Note: The factor is $\pi/L$, not $2\pi/L$ as in standard FFT.

### 6.3 Integration with Existing Code

The CFFT can be integrated as an alternative FFT backend:

```cpp
class CrysFFT : public FFT<double> {
    // Use DCT internally for forward/backward transforms
    // Expose same interface as standard FFT
    void forward(double* rdata, double* kdata) override;
    void backward(double* kdata, double* rdata) override;
};
```

### 6.4 Relationship with Reduced Basis

CFFT and SpaceGroup reduced basis are complementary:

| Feature | SpaceGroup Reduced Basis | CFFT |
|---------|-------------------------|------|
| Scope | Field storage | FFT computation |
| Reduction | Irreducible mesh (~96x for BCC) | Mirror symmetry (8-32x) |
| Applicability | All space groups | Centrosymmetric with 3 mirrors |
| Combination | Can be used together for maximum benefit |

---

## 7. References

1. Qiang, Y.; Li, W. "Accelerated Pseudo-Spectral Method of Self-Consistent Field Theory via Crystallographic Fast Fourier Transform" *Macromolecules* **2020**, 53, 9943-9952.

2. Ten Eyck, L. F. "Crystallographic Fast Fourier Transforms" *Acta Cryst.* **1973**, A29, 183-191.

3. Kudlicki, A.; Rowicka, M.; Otwinowski, Z. "The Crystallographic Fast Fourier Transform. Recursive Symmetry Reduction" *Acta Cryst.* **2007**, A63, 465-480.

---

## Glossary

| Term | Definition |
|------|------------|
| DCT | Discrete Cosine Transform |
| CFFT | Crystallographic Fast Fourier Transform |
| Asymmetric unit | Smallest region that generates the full unit cell by symmetry |
| Centrosymmetric | Having an inversion center (point where $\mathbf{r} \to -\mathbf{r}$) |
| Glide plane | Mirror + translation parallel to the plane |
