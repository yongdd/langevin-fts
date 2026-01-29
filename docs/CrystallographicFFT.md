# Crystallographic FFT (CrysFFT)

> **Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the details against the original references and the implementation before relying on this document for research or production use.

This document describes the crystallographic FFT (CrysFFT) paths used to accelerate pseudo-spectral diffusion when space-group symmetry is enabled. Two algorithms are implemented:

- **Pmmm DCT**: DCT-II/III on the 1/8 physical grid (mirror symmetry).
- **3m recursive (2x2x2)**: recursive crystallographic FFT using symmetry translations (Qiang and Li, 2020).

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
- **Physical grid**: symmetry-reduced grid used by CrysFFT, size `(Nx/2) x (Ny/2) x (Nz/2)`.
- **Irreducible grid**: smallest symmetry-unique grid (one point per symmetry orbit).
- **Reduced basis**: the storage basis used by `SpaceGroup` (irreducible, Pmmm physical, or m3 physical).

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

---

## 3. Activation and Selection

CrysFFT is enabled automatically when all of the following are true:

- `space_group` is set.
- Dimension is 3D.
- Boundary conditions are periodic.
- The computation box is orthogonal.
- `Nx`, `Ny`, `Nz` are even.
- Fields are real-valued (double precision).

Selection order:
1. **3m recursive** if the space group provides 3m translations and `Nz/2` satisfies the alignment constraint.
2. **Pmmm DCT** if the space group has mirror planes in x/y/z.
3. Otherwise, fall back to standard FFT.

CrysFFT is used in pseudo-spectral solvers for continuous (RQM4/RK2) and discrete chains (full and half-bond steps), and in the stress computation path when applicable.

---

## 4. Basis Choices and Mapping

CrysFFT operates on the physical grid, but `SpaceGroup` can store fields in different bases. The solver may need to map between the stored basis and the physical grid.

### 4.1 Irreducible basis (default)

The irreducible basis stores one representative per symmetry orbit. It is the\n+smallest storage basis produced by `SpaceGroup` and is independent of any\n+particular CrysFFT algorithm.\n+\n+- Stores only irreducible mesh points (one per symmetry orbit).\n+- Maximum memory reduction.\n+- Requires gather/scatter mapping between the irreducible basis and the\n+  physical grid when CrysFFT is used.

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
- Orthogonal boxes only.
- Even grid dimensions (`Nx`, `Ny`, `Nz` must be even).
- Real-valued fields only.
- **3m recursive** requires 3m translations and `Nz/2` divisible by 8.
- **Pmmm DCT** requires mirror planes in x, y, z.

---

## 7. Implementation Notes

### 7.1 CPU (FFTW)

- `FftwCrysFFT`: Pmmm DCT-II/III using FFTW REDFT10/REDFT01.
- `FftwCrysFFTRecursive3m`: recursive 3m algorithm with twiddle-factor caches.
- FFTW multithreading is disabled; concurrency is handled at the solver level (OpenMP over propagators).
- Thread-local buffers are used to avoid race conditions.

### 7.2 CUDA

- `CudaCrysFFT`: Pmmm DCT-II/III via `CudaRealTransform` (cuFFT-based).
- `CudaCrysFFTRecursive3m`: recursive 3m algorithm implemented on GPU, with stream-aware diffusion.
- The CUDA DCT path uses `CudaRealTransform` (see `docs/CudaRealTransform.md`).

---

## 8. Performance Notes

- Speedup depends on grid size, symmetry, and whether mapping is required.
- Automatic physical-basis selection avoids gather/scatter overhead when possible.
- Small grids may show limited speedup due to fixed overheads.
- The 3m recursive algorithm typically outperforms Pmmm DCT when available.

---

## 9. References

1. Qiang, Y.; Li, W. "Accelerated pseudo-spectral method of self-consistent field theory via crystallographic fast Fourier transform." *Macromolecules* **2020**, 53, 9943-9952.
2. Ten Eyck, L. F. "Crystallographic Fast Fourier Transforms." *Acta Cryst.* **1973**, A29, 183-191.
3. Kudlicki, A.; Rowicka, M.; Otwinowski, Z. "The Crystallographic Fast Fourier Transform. Recursive Symmetry Reduction." *Acta Cryst.* **2007**, A63, 465-480.
