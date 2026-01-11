# Features

This document provides a detailed overview of the features available in the polymer field theory simulation library.

## Core Library

### Polymer Architecture
- **Any number of monomer types**: Support for arbitrary multi-component systems
- **Arbitrary acyclic branched polymers**: Star, comb, dendritic, bottle-brush, and other branched architectures
- **Arbitrary mixtures**: Block copolymers, homopolymers, and their mixtures
- **Conformational asymmetry**: Different statistical segment lengths for different monomer types

### Chain Propagator Computation
- **Automatic optimization**: Dynamic programming to avoid redundant calculations for branched polymers
  - Arbitrary floating point block lengths for continuous chains (not limited to integer multiples of ds)
- **Access to chain propagators**: Direct access to propagator values at any contour step
- **Arbitrary initial conditions**: Custom initial conditions at chain ends (e.g., for grafted polymers)

### Chain Models
- **Continuous**: Gaussian chain model solving the modified diffusion equation
- **Discrete**: Freely-jointed chain model using Chapman-Kolmogorov equations

### Numerical Methods

#### Pseudo-Spectral Method
- 4th-order Richardson extrapolation for continuous chains
- Supports both continuous and discrete chain models
- Boundary conditions:
  - Periodic (FFT)
  - Reflecting (DCT) - zero flux at boundaries
  - Absorbing (DST) - zero value at boundaries

#### Real-Space Method
- Crank-Nicolson ADI (Alternating Direction Implicit) finite difference scheme
- 2nd-order (default) or 4th-order Richardson extrapolation (via `-DPOLYMERFTS_USE_RICHARDSON=ON`)
- Supports only continuous chain model
- Supports periodic, reflecting, and absorbing boundaries
- See [RealSpaceSolver.md](RealSpaceSolver.md) for details

### Simulation Box
- **Dimensions**: 1D, 2D, and 3D
- **Impenetrable regions**: Can set masked regions where polymers cannot enter
- **Space group symmetries**: Constrain field symmetries during SCFT iterations (**beta**)

### Platforms and Performance
- **CPU**: Intel MKL backend with OpenMP parallelization
- **GPU**: NVIDIA CUDA backend with multi-stream execution
- **Parallel propagator computation**:
  - CPU: Up to 8 cores
  - GPU: Up to 4 CUDA streams
- **Memory saving option**: Checkpoint-based propagator storage for large systems (CPU and CUDA)
- **Common interfaces**: Same API regardless of chain model, dimension, or platform

### Field Iteration
- **Anderson mixing**: Accelerated convergence for self-consistent field iterations

---

## SCFT, L-FTS, and CL-FTS Modules

High-level simulation modules built on top of the core library.

### Supported Systems
- Polymer melts
- Arbitrary mixtures of block copolymers, homopolymers, and random copolymers

### Self-Consistent Field Theory (SCFT)
- Box size determination by stress calculation
- Anderson mixing and ADAM optimizers
- Smearing for finite-range interactions

### Langevin Field-Theoretic Simulation (L-FTS)
- Leimkuhler-Matthews method for field updates
- PCG64 random number generator
- Structure function calculations
- Smearing for finite-range interactions

### Complex Langevin FTS (CL-FTS)
- Complex-valued auxiliary fields for handling the sign problem
- Dynamical stabilization option
- Smearing for finite-range interactions

---

## Feature Comparison

| Feature | Pseudo-Spectral | Real-Space |
|---------|-----------------|------------|
| Continuous chains | Yes | Yes |
| Discrete chains | Yes | No |
| Periodic BC | Yes (FFT) | Yes |
| Reflecting BC | Yes (DCT) | Yes |
| Absorbing BC | Yes (DST) | Yes |
| Temporal accuracy | 4th-order | 2nd or 4th-order |
| Stress calculation | Yes | No |
| Recommended for | Large grids, periodic systems | Non-periodic boundaries |

---

## References

For implementation details and algorithms, see:
- Chain propagator optimization: *J. Chem. Theory Comput.* **2025**, 21, 3676
- Discrete chain model: *J. Chem. Phys.* **2019**, 150, 234901
- Multi-monomer theory: *Macromolecules* **2025**, 58, 816
