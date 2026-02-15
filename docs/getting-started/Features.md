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
| **SCFT** | Mean-field equilibrium with Anderson mixing or ADAM optimizer, stress-based box optimization, and space groups. |
| **L-FTS** | Langevin dynamics (Leimkuhler-Matthews) with structure function calculations and Well-Tempered Metadynamics (WTMD). |
| **CL-FTS** (beta) | Complex Langevin with dynamical stabilization. |

---

## Advanced Features

| Feature | Description |
|---------|-------------|
| Branched polymers | Star, comb, dendritic, bottle-brush architectures |
| Random copolymers | Statistical copolymers with specified monomer fractions |
| Propagator optimization | Dynamic programming for redundant calculation elimination |
| Numerical methods | RQM4 (4th), RK2 (2nd), CN-ADI2 (2nd) for continuous chains |
| Custom initial conditions | Arbitrary initial conditions at chain ends (core library) |
| Impenetrable regions | Masked regions where polymers cannot enter (core library) |
| Boundary conditions | Periodic (FFT), reflecting (DCT), absorbing (DST) (core library) |
| Space groups | 230 space groups via spglib (beta) |
| CrysFFT | Symmetry-accelerated FFT: Pmmm (BCC), 3m (Gyroid), hexagonal (HCP, PL) (beta) |
| Non-orthogonal box | Monoclinic and triclinic unit cells with arbitrary angles |
| Crystal system constraints | Orthorhombic, tetragonal, cubic, hexagonal, monoclinic, triclinic |
| ADAM optimizer | Alternative to Anderson mixing for SCFT (SCFT) |
| Well-Tempered Metadynamics | Enhanced sampling for phase transitions (L-FTS) |
| Field compressors | Linear Response (LR), Anderson Mixing (AM), and hybrid (LRAM) for saddle point (L-FTS) |
| Smearing | Finite-range interactions for UV regularization (SCFT, L-FTS, CL-FTS) |
| Memory saving | Checkpoint-based storage for large systems |
| Parallel execution | OpenMP threads (CPU), multi-stream (CUDA) |

---

## References

- Chain propagator optimization: *J. Chem. Theory Comput.* **2025**, 21, 3676
- Discrete chain model: *J. Chem. Phys.* **2019**, 150, 234901
- RQM4 method: *Macromolecules* **2008**, 41, 942
