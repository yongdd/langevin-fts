# Features

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

## Overview

- **Core library**: C++/CUDA propagator computation with Python bindings. Unified API across chain models, dimensions, and platforms.
- **Simulation modules**: High-level Python classes (SCFT, L-FTS, CL-FTS) built on the core library.

| Category | Features |
|----------|----------|
| Polymers | Any monomer types, acyclic branched architectures, mixtures |
| Chain models | Continuous (Gaussian), Discrete (bead-spring) |
| Dimensions | 1D, 2D, 3D |
| Platforms | CPU (OpenMP), GPU (CUDA) |
| Simulations | SCFT, L-FTS, CL-FTS (beta) |

---

## Simulation Types

| Type | Description |
|------|-------------|
| **SCFT** | Mean-field equilibrium with Anderson mixing, stress-based box optimization, and space groups. |
| **L-FTS** | Langevin dynamics (Leimkuhler-Matthews) with structure function calculations. |
| **CL-FTS** (beta) | Complex Langevin with smearing and dynamical stabilization. |

---

## Advanced Features

| Feature | Description |
|---------|-------------|
| Branched polymers | Star, comb, dendritic, bottle-brush architectures |
| Propagator optimization | Dynamic programming for redundant calculation elimination |
| Numerical methods | RQM4 (4th), RK2 (2nd), CN-ADI2 (2nd) for continuous chains |
| Custom initial conditions | Arbitrary initial conditions at chain ends (core library) |
| Impenetrable regions | Masked regions where polymers cannot enter (core library) |
| Boundary conditions | Periodic (FFT), reflecting (DCT), absorbing (DST) (core library) |
| Space groups | 230 space groups via spglib (beta) |
| CrysFFT | Symmetry-accelerated FFT: Pmmm (BCC), 3m (Gyroid), hexagonal (HCP, PL) (beta) |
| Memory saving | Checkpoint-based storage for large systems |
| Parallel execution | OpenMP threads (CPU), multi-stream (CUDA) |

---

## References

- Chain propagator optimization: *J. Chem. Theory Comput.* **2025**, 21, 3676
- Discrete chain model: *J. Chem. Phys.* **2019**, 150, 234901
- RQM4 method: *Macromolecules* **2008**, 41, 942
