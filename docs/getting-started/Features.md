# Features

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
| **SCFT** | Mean-field equilibrium with Anderson mixing, stress-based box optimization, space groups, and the mean field of smeared Coulomb interactions with point ions and fixed charges in periodic or closed boxes (Poisson equation, Donnan potential). |
| **L-FTS** | Langevin dynamics (Leimkuhler-Matthews) with structure function calculations and Well-Tempered Metadynamics (WTMD). |
| **CL-FTS** (beta) | Complex Langevin with dynamical stabilization, smeared Coulomb interactions, optionally screened by a background electrolyte (charged polymers, point ions with automatic counterions, fixed charges, periodic or closed boxes), grand-canonical polymers and ions (Donnan potential), an exponential-time-differencing integrator with the exact linear response, chemical potential and osmotic pressure operators, and a Gibbs-ensemble driver for coexisting phases. |

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
| Space groups | All 230 space groups in the core `SpaceGroup` class (3D, via spglib); SCFT accepts the orthorhombic to cubic groups, numbers 16–230 (beta) |
| CrysFFT | Symmetry-accelerated FFT: 3m recursive (BCC, FCC, Gyroid, DD), Pmmm DCT as fallback for groups with x/y/z mirror planes, ObliqueZ z-mirror (HCP, PL) (beta) |
| Non-orthogonal box | Monoclinic and triclinic unit cells with arbitrary angles |
| Crystal system constraints | Orthorhombic, tetragonal, cubic, hexagonal, monoclinic, triclinic |
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
