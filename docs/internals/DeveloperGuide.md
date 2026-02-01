# Developer Documentation

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the class hierarchies, design patterns, and development practices in the polymer field theory simulation library.

## Table of Contents

1. [Overview](#1-overview)
2. [Abstract Factory Pattern](#2-abstract-factory-pattern)
3. [Core Class Hierarchies](#3-core-class-hierarchies)
4. [Template Instantiation Strategy](#4-template-instantiation-strategy)
5. [Development Workflow](#5-development-workflow)
6. [Adding New Features](#6-adding-new-features)
7. [Code Standards](#7-code-standards)
8. [Debugging Tips](#8-debugging-tips)
9. [Contributing Guidelines](#9-contributing-guidelines)
10. [References](#10-references)

---

## 1. Overview

The library uses the **Abstract Factory Pattern** to support multiple computational platforms (CPU, GPU) through a unified interface. All major computational classes are templates parameterized on:

- `T` = `double` or `std::complex<double>` for real or complex field theories
- `DIM` = 1, 2, or 3 for spatial dimension (FFT classes only)

### Directory Structure

```
src/
├── common/           # Platform-independent interfaces and utilities
├── platforms/
│   ├── cpu/          # CPU implementations (MKL, FFTW)
│   └── cuda/         # NVIDIA CUDA implementations
├── python/           # Python interface modules
└── pybind11/         # C++/Python bindings
```

---

## 2. Abstract Factory Pattern

**Location**: `src/common/AbstractFactory.h`

The abstract factory provides a unified interface for creating platform-specific objects without exposing implementation details.

### Class Diagram

```
AbstractFactory<T>                    [Abstract Base]
├── create_computation_box()          [pure virtual]
├── create_molecules_information()    [pure virtual]
├── create_propagator_computation()   [pure virtual]
├── create_anderson_mixing()          [pure virtual]
└── display_info()                    [pure virtual]
        │
        ├── FftwFactory<T>             [CPU Implementation]
        │   └── Creates: CpuComputationBox, CpuComputation*,
        │                CpuAndersonMixing, CpuSolver*
        │
        └── CudaFactory<T>            [GPU Implementation]
            └── Creates: CudaComputationBox, CudaComputation*,
                         CudaAndersonMixing*, CudaSolver*
```

### Platform Selection

```cpp
#include "PlatformSelector.h"

// Create factory for specific platform
auto factory = PlatformSelector::create_factory("cuda", reduce_memory);

// Or use automatic selection (GPU for 2D/3D, CPU for 1D)
auto factory = PlatformSelector::create_factory("auto", reduce_memory);
```

### Factory Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `create_computation_box(nx, lx, bc)` | `ComputationBox<T>*` | Simulation grid with boundary conditions |
| `create_computation_box(nx, lx, bc, mask)` | `ComputationBox<T>*` | Grid with impenetrable mask regions |
| `create_molecules_information(model, ds, bond_lengths)` | `Molecules*` | Polymer/solvent container |
| `create_propagator_computation(box, molecules, optimizer, method)` | `PropagatorComputation<T>*` | Propagator solver (method: rqm4, rk2, cn-adi2) |
| `create_anderson_mixing(n_var, max_hist, ...)` | `AndersonMixing<T>*` | Field update accelerator |

---

## 3. Core Class Hierarchies

### 3.1 ComputationBox

**Location**: `src/common/ComputationBox.h`

Manages spatial discretization, boundary conditions, and grid operations.

```
ComputationBox<T>                     [Base Class]
├── Members:
│   ├── nx[3]           - grid points per dimension
│   ├── lx[3]           - box lengths
│   ├── dx[3]           - grid spacing
│   ├── bc[6]           - boundary conditions
│   └── mask            - impenetrable regions (optional)
├── Methods:
│   ├── integral(g)              - ∫g(r)dr
│   ├── inner_product(g, h)      - ∫g(r)h(r)dr
│   ├── multi_inner_product()    - sum of inner products
│   ├── zero_mean(g)             - remove spatial mean
│   ├── set_lx(lx)               - update box lengths [virtual]
│   └── set_lattice_parameters() - update with angles [virtual]
        │
        ├── CpuComputationBox<T>      [CPU Memory]
        │
        └── CudaComputationBox<T>     [GPU Memory]
```

**Boundary Conditions:**

| Type | FFT Transform | Physical Meaning |
|------|---------------|------------------|
| `periodic` | Complex FFT | Periodic boundaries |
| `reflecting` | DCT (Type II/III) | Neumann (zero flux) |
| `absorbing` | DST (Type II/III) | Dirichlet (zero value) |

### 3.2 PropagatorComputation

**Location**: `src/common/PropagatorComputation.h`

Central computational engine for solving modified diffusion equations and computing polymer statistics.

```
PropagatorComputation<T>                      [Abstract Base]
├── compute_propagators(w, q_init)            [pure virtual]
├── compute_concentrations()                  [pure virtual]
├── compute_stress()                          [pure virtual]
├── get_total_partition(polymer)              [pure virtual]
├── get_chain_propagator(p, v, u, n)          [pure virtual]
├── get_total_concentration(monomer)          [pure virtual]
└── update_laplacian_operator()               [pure virtual]
        │
        ├── CpuComputationBase<T>             [CPU Common Logic]
        │   │
        │   ├── CpuComputationContinuous<T>   [Continuous Chains]
        │   │   └── RQM4, RK2, or CN-ADI2 (selectable)
        │   │
        │   └── CpuComputationDiscrete<T>     [Discrete Chains]
        │       └── Boltzmann weight + bond convolution
        │
        └── CudaComputationBase<T>            [GPU Common Logic]
            │
            ├── CudaComputationContinuous<T>
            ├── CudaComputationDiscrete<T>
            ├── CudaComputationReduceMemoryContinuous<T>
            └── CudaComputationReduceMemoryDiscrete<T>
```

**Chain Models:**

| Model | Propagator Equation | Solver Method |
|-------|---------------------|---------------|
| Continuous | $\partial q/\partial s = (b^2/6)\nabla^2 q - wq$ | RQM4, RK2, or CN-ADI2 |
| Discrete | Recursive integral (Chapman-Kolmogorov) | Pseudo-spectral (bond convolution) |

### 3.3 Solver Hierarchy

**Location**: `src/platforms/cpu/CpuSolver.h`, `src/platforms/cuda/CudaSolver.h`

```
CpuSolver<T>                                  [Abstract Base]
├── exp_dw[monomer]         - Boltzmann factors
├── exp_dw_half[monomer]    - Half-step Boltzmann factors
├── update_dw(w)            [pure virtual]
├── advance_propagator()    [pure virtual]
└── compute_single_segment_stress() [pure virtual]
        │
        ├── CpuSolverPseudoBase<T>            [Pseudo-spectral Common]
        │   ├── fft_               - FFT<T>* for transforms
        │   ├── transform_forward()
        │   ├── transform_backward()
        │   └── update_laplacian_operator()
        │       │
        │       ├── CpuSolverPseudoRQM4<T>
        │       │   ├── update_dw(): exp(-w·ds/2)
        │       │   └── advance_propagator(): Richardson
        │       │
        │       ├── CpuSolverPseudoRK2<T>
        │       │   ├── update_dw(): exp(-w·ds/2)
        │       │   └── advance_propagator(): single RK step
        │       │
        │       └── CpuSolverPseudoDiscrete<T>
        │           ├── update_dw(): exp(-w·ds)
        │           ├── advance_propagator(): bond convolution
        │           └── advance_propagator_half_bond_step()
        │
        └── CpuSolverCNADI<T>                  [Finite Difference]
            └── CN-ADI with tridiagonal solver
```

### 3.4 FFT Hierarchy

**Location**: `src/platforms/cpu/FFT.h`, `src/platforms/cpu/FftwFFT.h`

```
FFT<T>                                        [Abstract Base]
├── forward(T* real, double* spectral)        [pure virtual]
├── backward(double* spectral, T* real)       [pure virtual]
├── forward(T* real, complex* spectral)       [pure virtual, periodic only]
└── backward(complex* spectral, T* real)      [pure virtual, periodic only]
        │
        ├── FftwFFT<T, DIM>                    [CPU]
        │   └── DIM = 1, 2, or 3 (compile-time)
        │   └── Uses MKL or FFTW backend
        │
        └── CudaFFT<T, DIM>                   [GPU/cuFFT + CudaRealTransform]
            ├── forward_stream(..., cudaStream_t)
            ├── backward_stream(..., cudaStream_t)
            └── Uses CudaRealTransform for non-periodic BC (DCT/DST)
```

**Transform Types:**

| Boundary | Forward | Backward | Notes |
|----------|---------|----------|-------|
| Periodic | FFT | IFFT | Complex coefficients |
| Reflecting | DCT-II | DCT-III | Real coefficients |
| Absorbing | DST-II | DST-III | Real coefficients |

### 3.5 AndersonMixing Hierarchy

**Location**: `src/common/AndersonMixing.h`

```
AndersonMixing<T>                             [Abstract Base]
├── n_var           - number of field variables
├── max_hist        - maximum history length
├── mix_min/mix_init - mixing parameter bounds
├── calculate_new_fields(w_new, w_current, w_deriv) [pure virtual]
└── reset_count()   [pure virtual]
        │
        ├── CpuAndersonMixing<T>              [CPU Implementation]
        │   └── Uses CircularBuffer for history
        │
        ├── CudaAndersonMixing<T>             [GPU, Standard]
        │   └── Uses CudaCircularBuffer (device memory)
        │
        └── CudaAndersonMixingReduceMemory<T> [GPU, Memory-saving]
            └── Uses PinnedCircularBuffer (host memory)
```

### 3.6 Molecules and Polymer

**Location**: `src/common/Polymer.h`, `src/common/Molecules.h`

Polymers are represented as **acyclic graphs** (trees):
- **Vertices**: Junction points and chain ends
- **Edges**: Polymer blocks connecting vertices

```
Block                                         [Block Specification]
├── monomer_type    - e.g., "A", "B"
├── contour_length  - relative length (0 to 1)
├── v, u            - connecting vertices
└── n_segment       - discretized segment count

Polymer                                       [Single Chain Type]
├── alpha           - total contour length
├── volume_fraction - system fraction
├── blocks[]        - vector of Block
├── adjacent_nodes  - graph adjacency list
└── Methods:
    ├── get_block(i), get_n_blocks()
    ├── get_adjacent_nodes(v)
    └── display_architecture()

Molecules                                     [System Container]
├── model_name      - "continuous" or "discrete"
├── ds              - contour step size
├── bond_lengths    - {monomer: (a/a_ref)²}
├── polymer_types[] - vector of Polymer
└── Methods:
    ├── add_polymer(phi, blocks)
    ├── add_polymer(phi, blocks, grafting)
    ├── add_solvent(phi, monomer)
    └── get_polymer(i), get_n_polymer_types()
```

**Example: ABC Triblock (linear)**

```
    0 ----A---- 1 ----B---- 2 ----C---- 3

    blocks = [
        {"type": "A", "length": 0.33},
        {"type": "B", "length": 0.34},
        {"type": "C", "length": 0.33}
    ]
```

**Example: Star Polymer (3-arm)**

```
           1
           |
           A
           |
    2--B-- 0 --C-- 3

    blocks = [
        {"type": "A", "length": 0.33, "v": 0, "u": 1},
        {"type": "B", "length": 0.33, "v": 0, "u": 2},
        {"type": "C", "length": 0.34, "v": 0, "u": 3}
    ]
```

---

## 4. Template Instantiation Strategy

All computational classes use explicit template instantiation for:
- Faster compilation (templates compiled once)
- Smaller binary size
- Better error messages

**Location**: `src/common/TemplateInstantiations.h`

```cpp
// For single-parameter templates (most classes)
INSTANTIATE_CLASS(CpuComputationContinuous);
// Expands to:
//   template class CpuComputationContinuous<double>;
//   template class CpuComputationContinuous<std::complex<double>>;

// For FFT classes with dimension parameter
INSTANTIATE_FFT_CLASS(FftwFFT);
// Expands to:
//   template class FftwFFT<double, 1>;
//   template class FftwFFT<double, 2>;
//   template class FftwFFT<double, 3>;
//   template class FftwFFT<std::complex<double>, 1>;
//   template class FftwFFT<std::complex<double>, 2>;
//   template class FftwFFT<std::complex<double>, 3>;
```

---

## 5. Development Workflow

### Building and Testing

#### Rebuilding After C++ Changes

Changes to `src/common/*.cpp` or `src/platforms/*/*.cpp|.cu` require rebuilding:

```bash
cd build && make -j8 && make install
```

#### Python Changes

Changes to `src/python/*.py` take effect after `make install` from build directory. No recompilation needed.

#### Running Tests

See [Installation.md](../getting-started/Installation.md#testing) for test commands.

---

## 6. Adding New Features

### Platform Implementation

When adding new computational features, you must implement them for both platforms unless explicitly platform-specific:
- `src/platforms/cpu/` - CPU implementations (MKL, FFTW)
- `src/platforms/cuda/` - CUDA GPU implementations

### Memory Management

C++ code uses raw pointers. Ensure proper allocation/deallocation in constructors/destructors.

### Propagator Computation Optimizer

The `PropagatorComputationOptimizer` automatically detects redundant calculations using hash tables of `PropagatorCode` objects. Avoid manual optimization.

### New Monomer Types or Interactions

Modify only the parameter dictionary - the code supports arbitrary numbers of monomer types. The `SymmetricPolymerTheory` class handles interaction matrix eigendecomposition automatically.

### New Numerical Methods

1. Implement the solver in both `cpu/` and `cuda/` directories
2. Add the method to the factory classes
3. Update parameter validation in Python modules
4. Add tests and benchmarks

---

## 7. Code Standards

### C++

- Use C++20 features where appropriate
- Follow existing naming conventions
- Document public APIs

### Python

- Follow PEP 8 style guidelines
- Add docstrings for public functions
- Include type hints where practical

---

## 8. Debugging Tips

### Segmentation Faults

See [Installation.md](../getting-started/Installation.md#segmentation-fault) for stack size settings.

### CUDA Debugging

- Use `cuda-memcheck` for memory errors
- Check return values of CUDA API calls
- Use `nvprof` or Nsight for profiling

### Platform Consistency Issues

If GPU and CPU give different results:
1. Check floating-point operation order
2. Verify FFT normalization
3. Check memory alignment

### Verifying Numerical Accuracy

1. Run same test on both `cuda` and CPU platforms - results should match to ~10⁻¹³
2. Check partition function consistency: `solver.check_total_partition()` should pass
3. Verify material conservation: `np.mean(sum(phi_species))` = 1.0 within machine precision (~10⁻¹⁵)

---

## 9. Contributing Guidelines

### What We Accept

- Python scripts for specific polymer morphologies
- Modified versions of `scft.py`, `lfts.py`, etc.
- Bug fixes and performance improvements

### Requirements

- Contributions should contain sample results, test code, or desired outputs
- There should be relevant published literature
- Code does not have to be optimal or excellent
- **Contributed code must not contain GPL-licensed code**

### What to Avoid

- GPL-licensed code (due to licensing conflicts)

### Submitting Contributions

- Open a pull request on GitHub
- For C++/CUDA changes you find difficult to implement, open an issue with sample code in any language

**Note**: This library may be updated without backward compatibility. Contributed code will be maintained to work with new versions.

---

## 10. References

### Chain Propagator Computation
- D. Yong and J. U. Kim, "Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations," *J. Chem. Theory Comput.* **2025**, 21, 3676.

### Discrete Chain Model
- S. J. Park, D. Yong, Y. Kim, and J. U. Kim, "Numerical implementation of pseudo-spectral method in self-consistent mean field theory for discrete polymer chains," *J. Chem. Phys.* **2019**, 150, 234901.

### Multi-Monomer Polymer Field Theory
- D. Morse, D. Yong, and K. Chen, "Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems," *Macromolecules* **2025**, 58, 816.

### CUDA Implementation
- G. K. Cheong, A. Chawla, D. C. Morse, and K. D. Dorfman, "Open-source code for self-consistent field theory calculations of block polymer phase behavior on graphics processing units," *Eur. Phys. J. E* **2020**, 43, 15.
- D. Yong, Y. Kim, S. Jo, D. Y. Ryu, and J. U. Kim, "Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces," *Macromolecules* **2021**, 54, 11304.

### Field Update Algorithms
- B. Vorselaars, "Efficient Langevin and Monte Carlo sampling algorithms: the case of field-theoretic simulations," *J. Chem. Phys.* **2023**, 158, 114117.
- A. Arora, D. C. Morse, F. S. Bates, and K. D. Dorfman, "Accelerating self-consistent field theory of block polymers in a variable unit cell," *J. Chem. Phys.* **2017**, 146, 244902.
