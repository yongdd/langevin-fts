# Crystallographic FFT (CrysFFT)

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the crystallographic FFT (CrysFFT) paths used to accelerate pseudo-spectral diffusion when space-group symmetry is enabled. Three algorithms are implemented:

- **Pmmm DCT**: DCT-II/III on the 1/8 physical grid (mirror symmetry).
- **3m recursive (2x2x2)**: recursive crystallographic FFT using symmetry translations (Qiang and Li, 2020).
- **Hex z-mirror**: DCT-z + FFT-xy on the z-mirror physical grid for hexagonal cells.

**Status:** Implemented on CPU (FFTW) and CUDA.

## Table of Contents

1. [Overview](#1-overview)
2. [Algorithms](#2-algorithms)
3. [Activation and Selection](#3-activation-and-selection)
4. [Basis Choices and Mapping](#4-basis-choices-and-mapping)
5. [Usage](#5-usage)
6. [Requirements and Limitations](#6-requirements-and-limitations)
7. [Implementation Notes](#7-implementation-notes)
8. [Performance Notes](#8-performance-notes)
9. [References](#9-references)

---

## 1. Overview

CrysFFT replaces the standard 3D FFT in pseudo-spectral propagator updates with symmetry-reduced transforms whenever a compatible space group is enabled.

Terminology:
- **Logical grid**: full simulation grid, size `Nx x Ny x Nz`.
- **Physical grid**: CrysFFT working grid.
  - Pmmm / 3m: `(Nx/2) x (Ny/2) x (Nz/2)` (mirror-reduced).
  - Hex z-mirror: `Nx x Ny x (Nz/2)` (z-mirror only).
- **Irreducible grid**: smallest symmetry-unique grid (one point per symmetry orbit).
- **Reduced basis**: storage basis used by `SpaceGroup` (irreducible, Pmmm physical, M3 physical, or z-mirror physical).

CrysFFT is used only for diffusion steps in pseudo-spectral solvers. All real-space operations (e.g., multiplication by `exp(-w ds/2)`) remain unchanged.

---

## 2. Algorithms

### 2.1 Pmmm DCT (mirror symmetry)

For space groups with three perpendicular mirror planes (Pmmm), the diffusion step uses DCT-II/III on the physical grid:

```
q_out = DCT-III[ exp(-k^2 * coeff) * DCT-II[q_in] ]
```

- The transform is real-to-real (DCT-II forward, DCT-III inverse).
- The physical grid is `(Nx/2) x (Ny/2) x (Nz/2)`.
- Wavenumbers use `k = pi * i / L` along each axis.

### 2.2 3m recursive (2x2x2) algorithm

For space groups with 3m translations (mirror symmetry plus specific translational offsets), the recursive 2x2x2 algorithm reduces the transform further by exploiting translational symmetry. This is the CrysFFT described by Qiang and Li (2020).

Key features:
- Operates on the same physical grid size `(Nx/2) x (Ny/2) x (Nz/2)`.
- Uses eight sub-transforms with precomputed twiddle factors derived from symmetry translations.
- Reduces arithmetic by avoiding redundant mirror/translation work.

Both CPU and CUDA implementations are provided (`FftwCrysFFTRecursive3m` and `CudaCrysFFTRecursive3m`).

### 2.3 Hex z-mirror (DCT-z + FFT-xy)

For hexagonal space groups with a pure z-mirror (translation `t_z = 0`), the diffusion step uses:

```
q_out = DCT-III_z[ exp(-k^2 * coeff) * FFT_xy[ DCT-II_z[q_in] ] ]
```

- The physical grid is `Nx x Ny x (Nz/2)` (mirror in z only).
- The in-plane FFT uses the **hexagonal reciprocal metric** (`γ = 120°`).
- Only the pure mirror case (`t_z = 0`) is supported; the glide case (`t_z = 1/2`) is not yet implemented.

---

## 3. Activation and Selection

CrysFFT is enabled automatically when all of the following are true:

- `space_group` is set.
- Dimension is 3D.
- Boundary conditions are periodic.
- The computation box is orthogonal.
- `Nx`, `Ny`, `Nz` are even.
- Fields are real-valued (double precision).

Selection order (auto):
1. **3m recursive** if the space group provides 3m translations *and* `Nz/2 % 8 == 0`.
2. **Pmmm DCT** if the space group has mirror planes in x/y/z.
3. **Hex z-mirror** if the space group has a z-mirror with `t_z = 0` and the cell is hexagonal (`γ = 120°`).
4. Otherwise, fall back to standard FFT.

If a physical basis is forced in `SpaceGroup`, the selector respects it:
- **Pmmm physical basis forced** → Pmmm DCT (if mirror planes exist).
- **M3 physical basis forced** → 3m recursive (if translations and alignment are valid).
- **Z-mirror physical basis forced** → Hex z-mirror (only if `t_z = 0` and hexagonal cell); otherwise CrysFFT is disabled.

CrysFFT is used in pseudo-spectral solvers for continuous (RQM4/RK2) and discrete chains (full and half-bond steps), and in the stress computation path when applicable.

---

## 4. Basis Choices and Mapping

CrysFFT operates on the physical grid, but `SpaceGroup` can store fields in different bases. The solver may need to map between the stored basis and the physical grid.

### 4.1 Irreducible basis (default)

The irreducible basis stores one representative per symmetry orbit. It is the
smallest storage basis produced by `SpaceGroup` and is independent of any
particular CrysFFT algorithm.

- Stores only irreducible mesh points (one per symmetry orbit).
- Maximum memory reduction.
- Requires gather/scatter mapping between the irreducible basis and the
  physical grid when CrysFFT is used.

### 4.2 Pmmm physical basis

- Stores the physical grid for Pmmm symmetry: `(Nx/2) x (Ny/2) x (Nz/2)`.
- Mapping becomes identity (no gather/scatter).
- Available when the space group has mirror planes in x/y/z and the grid is even.

### 4.3 M3 physical basis

- Stores the even-index physical grid used by the 3m algorithm.
- Mapping becomes identity (no gather/scatter).
- Available when the space group provides 3m translations and the grid is even.

The solver automatically selects the fastest compatible physical basis when
space-group symmetry is enabled.

### 4.4 Z-mirror physical basis (hex)

- Stores the physical grid for z-mirror symmetry: `Nx x Ny x (Nz/2)`.
- Mapping becomes identity (no gather/scatter) when z-mirror is enabled.
- For `t_z = 1/2`, the basis uses a shifted z-slab (requires `Nz % 4 == 0`), but
  CrysFFT is not available until the glide-mirror variant is implemented.

**Optimizer note:** the field optimizer always uses the **irreducible** grid.
When a physical basis is active, the solver maps reduced ↔ irreducible
internally for optimizer-related operations.

---

## 5. Usage

### 5.1 SCFT parameters

```python
params = {
    "nx": [64, 64, 64],
    "lx": [4.0, 4.0, 4.0],
    "space_group": {
        "symbol": "Ia-3d",
        "number": 530,
    },
}
```

### 5.2 PropagatorSolver API

```python
from polymerfts.propagator_solver import PropagatorSolver

solver = PropagatorSolver(
    cb,
    molecules,
    space_group=sg,
)
```

The solver selects `m3-physical` when possible, otherwise `pmmm-physical`, and
falls back to the irreducible basis if neither is available.

---

## 6. Requirements and Limitations

- 3D only.
- Periodic boundary conditions only.
- Orthogonal boxes required for Pmmm / 3m.
- Hex z-mirror supports hexagonal cells (`γ = 120°`) with `t_z = 0`.
- Even `Nz` required for z-mirror; even `Nx, Ny, Nz` required for Pmmm / 3m.
- Real-valued fields only.
- **3m recursive** requires 3m translations and `Nz/2 % 8 == 0`.
- **Pmmm DCT** requires mirror planes in x, y, z.
- **Hex z-mirror** currently supports only `t_z = 0` (pure mirror), not the glide case.
- Non‑orthogonal boxes other than hex (`γ=120°`) disable CrysFFT.

---

## 7. Implementation Notes

### 7.1 CPU (FFTW)

- `FftwCrysFFTPmmm`: Pmmm DCT-II/III using FFTW REDFT10/REDFT01.
- `FftwCrysFFTRecursive3m`: recursive 3m algorithm with twiddle-factor caches.
- `FftwCrysFFTHex`: DCT-z + FFT-xy using the hexagonal reciprocal metric.
- FFTW multithreading is disabled; concurrency is handled at the solver level (OpenMP over propagators).
- Thread-local buffers are used to avoid race conditions.

### 7.2 CUDA

- `CudaCrysFFT`: Pmmm DCT-II/III via `CudaRealTransform` (cuFFT-based).
- `CudaCrysFFTRecursive3m`: recursive 3m algorithm implemented on GPU, with stream-aware diffusion.
- `CudaCrysFFTHex`: DCT-z + FFT-xy using cuFFT and hexagonal reciprocal metric.
- The CUDA DCT path uses `CudaRealTransform` (see `docs/CudaRealTransform.md`).

---

## 8. Performance Notes

- Speedup depends on grid size, symmetry, and whether mapping is required.
- Automatic physical-basis selection avoids gather/scatter overhead when possible.
- Small grids may show limited speedup due to fixed overheads.
- The 3m recursive algorithm typically outperforms Pmmm DCT when available.

### 8.1 64<sup>3</sup> FFT-only benchmark (Im-3m, BCC fields)

Benchmark setup (FFT-only diffusion step):
- `nx = [64, 64, 64]`, `lx = [4.0, 4.0, 4.0]`
- `coeff = 0.01` (diffusion coefficient; corresponds to `b^2 ds / 6`)
- 200 iterations, 20 warm-up iterations
- GPU: NVIDIA A10 (CUDA), `CUDA_VISIBLE_DEVICES=2`
- “off” uses full grid (no space group), “m3” and “pmmm” use the physical grids.
- Executable: `build/tests/TestCrysFFTCudaBench`

Results (ms/iter):

| Platform | off | m3-physical | pmmm-physical |
| --- | ---: | ---: | ---: |
| CUDA (A10) | 0.1750 | 0.0561 (3.12×) | 0.0893 (1.96×) |
| CPU (FFTW) | 1.9953 | 0.4101 (4.86×) | 0.5789 (3.45×) |

Notes:
- Speedups are computed relative to “off” on the same platform.
- The Pmmm result is forced even when 3m is available to enable direct comparison.

### 8.2 64<sup>3</sup> FFT-only benchmark (Hexagonal, P6/mmm)

Benchmark setup (FFT-only diffusion step):
- `nx = [64, 64, 64]`, `lx = [4.0, 4.0, 4.0]`, `γ = 120°`
- `coeff = 0.01`
- 200 iterations, 20 warm-up iterations
- GPU: NVIDIA A10 (CUDA), `CUDA_VISIBLE_DEVICES=2`
- Hex uses z-mirror physical grid (`Nx x Ny x Nz/2`); “off” uses full FFT.

Results (ms/iter):

| Platform | off | hex (DCT-z + FFT-xy) |
| --- | ---: | ---: |
| CUDA (A10) | 0.1291 | 0.1153 (1.12×) |
| CPU (FFTW) | 1.9517 | 1.9265 (1.01×) |

Notes:
- Hex speedups are relative to “off” on the same platform.
- On this grid, hex is not faster on CUDA; benefits may appear on larger grids.

### 8.3 Phase support (example grids)

The table below summarizes CrysFFT availability for the default phase grids used
in `examples/scft/phases/*.py` (same set as `tests/TestSpaceGroupPhases.py`).
Availability depends on the space group *and* grid constraints (3D, even `nx`,
orthogonal box, periodic BCs, and for 3m also the translation constraints).

| Phase | Space group | M3 recursive | Pmmm DCT | Hex z-mirror |
| --- | --- | :---: | :---: | :---: |
| BCC | Im-3m (529) | ✓ | ✓ |
| FCC | Fm-3m (523) | ✓ | ✓ |
| SC | Pm-3m (517) | ✓ | ✓ |
| A15 | Pm-3n (520) | ✓ | ✓ |
| DG | Ia-3d (530) | ✓ | ✗ |
| DD | Pn-3m (522) | ✓ | ✗ |
| DP | Im-3m (529) | ✓ | ✓ |
| SD | Fd-3m (526) | ✓ | ✗ |
| SG | I4_132 (510) | ✗ | ✗ |
| SP | Pm-3m (517) | ✓ | ✓ |
| Sigma | P4_2/mnm (419) | ✓ | ✗ |
| Fddd | Fddd (336) | ✓ | ✗ |
| HCP | P6_3/mmc (488) | ✗ | ✗ | ✗ (glide mirror) |
| PL | P6/mmm (485) | ✗ | ✗ | ✓ |

Notes:
- When both are available, the solver prefers **M3 recursive**.
- “✗” means CrysFFT is unavailable for that phase/grid; the solver uses the
  irreducible basis with standard FFT.
- **SG (I4_132)**: no mirror planes and no 3m translations → neither Pmmm nor M3 applies.
- **HCP/PL (hexagonal cells)**: non‑orthogonal boxes (γ=120°) → CrysFFT disabled.

---

## 9. References

1. Qiang, Y.; Li, W. "Accelerated pseudo-spectral method of self-consistent field theory via crystallographic fast Fourier transform." *Macromolecules* **2020**, 53, 9943-9952.
2. Ten Eyck, L. F. "Crystallographic Fast Fourier Transforms." *Acta Cryst.* **1973**, A29, 183-191.
3. Kudlicki, A.; Rowicka, M.; Otwinowski, Z. "The Crystallographic Fast Fourier Transform. Recursive Symmetry Reduction." *Acta Cryst.* **2007**, A63, 465-480.
