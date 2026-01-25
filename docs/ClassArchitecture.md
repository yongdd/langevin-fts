# Class Architecture and Design

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the class hierarchies, inheritance patterns, and design decisions in the polymer field theory simulation library.

## Table of Contents

1. [Overview](#overview)
2. [Abstract Factory Pattern](#abstract-factory-pattern)
3. [ComputationBox Hierarchy](#computationbox-hierarchy)
4. [PropagatorComputation Hierarchy](#propagatorcomputation-hierarchy)
5. [Solver Hierarchy](#solver-hierarchy)
6. [FFT Hierarchy](#fft-hierarchy)
7. [AndersonMixing Hierarchy](#andersonmixing-hierarchy)
8. [Molecules and Polymer Classes](#molecules-and-polymer-classes)
9. [Template Instantiation Strategy](#template-instantiation-strategy)
10. [Usage Examples](#usage-examples)

---

## Overview

The library uses the **Abstract Factory Pattern** to support multiple computational platforms (CPU with FFTW, GPU with CUDA) through a unified interface. All major computational classes are templates parameterized on:

- `T` = `double` or `std::complex<double>` for real or complex field theories
- `DIM` = 1, 2, or 3 for spatial dimension (FFT classes only)

### Directory Structure

```
src/
├── common/           # Platform-independent interfaces and utilities
├── platforms/
│   ├── cpu/          # FFTW implementations
│   └── cuda/         # NVIDIA CUDA implementations
├── python/           # Python interface modules
└── pybind11/         # C++/Python bindings
```

---

## Abstract Factory Pattern

**Location**: `src/common/AbstractFactory.h`

The abstract factory provides a unified interface for creating platform-specific objects without exposing implementation details.

### Class Diagram

```
AbstractFactory<T>                    [Abstract Base]
├── create_computation_box()          [pure virtual]
├── create_molecules_information()    [pure virtual]
├── create_propagator_computation()        [pure virtual]
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

// Or use automatic selection (CUDA for 2D/3D, FFTW for 1D)
auto factory = PlatformSelector::create_factory("auto", reduce_memory);
```

### Factory Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `create_computation_box(nx, lx, bc)` | `ComputationBox<T>*` | Simulation grid with boundary conditions |
| `create_computation_box(nx, lx, bc, mask)` | `ComputationBox<T>*` | Grid with impenetrable mask regions |
| `create_molecules_information(model, ds, bond_lengths)` | `Molecules*` | Polymer/solvent container |
| `create_propagator_computation(box, molecules, optimizer, method)` | `PropagatorComputation<T>*` | Propagator solver (method: rqm4, rk2, etdrk4, cn-adi2, cn-adi4-lr) |
| `create_anderson_mixing(n_var, max_hist, ...)` | `AndersonMixing<T>*` | Field update accelerator |

---

## ComputationBox Hierarchy

**Location**: `src/common/ComputationBox.h`

Manages spatial discretization, boundary conditions, and grid operations.

### Class Diagram

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

### Boundary Conditions

| Type | FFT Transform | Physical Meaning |
|------|---------------|------------------|
| `periodic` | Complex FFT | Periodic boundaries |
| `reflecting` | DCT (Type II/III) | Neumann (zero flux) |
| `absorbing` | DST (Type II/III) | Dirichlet (zero value) |

### Crystal Systems

The box supports non-orthogonal lattices via lattice vectors:

```cpp
// Orthogonal box (default)
box->set_lx({4.0, 4.0, 4.0});

// Non-orthogonal (e.g., hexagonal)
box->set_lattice_parameters({4.0, 4.0, 4.0}, {90.0, 90.0, 120.0});
```

---

## PropagatorComputation Hierarchy

**Location**: `src/common/PropagatorComputation.h`

Central computational engine for solving modified diffusion equations and computing polymer statistics.

### Class Diagram

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
        │   │   └── 4th-order Richardson extrapolation
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

### Chain Models

| Model | Propagator Equation | Solver Method |
|-------|---------------------|---------------|
| Continuous | $\partial q/\partial s = (b^2/6)\nabla^2 q - wq$ | Richardson extrapolation |
| Discrete | Recursive integral (Chapman-Kolmogorov) | Pseudo-spectral (bond convolution) |

#### Continuous Chain Model

For continuous Gaussian chains, the propagator satisfies the modified diffusion equation:

$$\frac{\partial q}{\partial s} = \frac{b^2}{6}\nabla^2 q - w \cdot q$$

This is solved using 4th-order Richardson extrapolation with the operator splitting method.

#### Discrete Chain Model Details

This library implements the discrete chain model as described in Park et al. (2019). Unlike continuous chains that solve a differential equation, discrete chains use **recursive integral equations** based on the Chapman-Kolmogorov equation.

**Units and Conventions:**
- **Unit length**: $bN^{1/2}$, where $b$ is the reference statistical segment length and $N$ is the reference polymerization index
- **Contour step size**: $\Delta s = 1/N$
- **Segment positions**: $s = \Delta s, 2\Delta s, \ldots, 1$ ($N$ segments total)

```
Discrete Chain Model:

    segment    segment    segment              segment
       1          2          3        ...         N
       |          |          |                    |
       s=Δs      s=2Δs     s=3Δs               s=1

    - N segments connected by N-1 bonds
    - Contour step: Δs = 1/N
    - Segment positions: s = Δs, 2Δs, 3Δs, ..., 1
```

**Propagator Evolution (N-1 Bond Model):**

The propagator evolution from segment $i$ to $i+1$ follows two steps:

1. **Full-segment Boltzmann weight:**
$$q^*(\mathbf{r}) = \exp(-w(\mathbf{r}) \cdot \Delta s) \cdot q_i(\mathbf{r})$$

2. **Bond convolution:**
$$q_{i+1}(\mathbf{r}) = \int g(\mathbf{R}) \, q^*(\mathbf{r} - \mathbf{R}) \, d\mathbf{R}$$

with initial condition: $q_1(\mathbf{r}) = \exp(-w(\mathbf{r}) \cdot \Delta s)$

**Bond Function $g(\mathbf{R})$:**

For the bead-spring (Gaussian) model:

$$g(\mathbf{R}) = \left(\frac{3}{2\pi b^2 \Delta s}\right)^{3/2} \exp\left(-\frac{3|\mathbf{R}|^2}{2 b^2 \Delta s}\right)$$

**Pseudo-Spectral Implementation:**

The bond convolution is computed efficiently in Fourier space:

$$\hat{q}_{i+1}(\mathbf{k}) = \hat{g}(\mathbf{k}) \cdot \hat{q}^*(\mathbf{k})$$

where the Fourier transform of the bond function is:

$$\hat{g}(\mathbf{k}) = \exp\left(-\frac{b^2 |\mathbf{k}|^2 \Delta s}{6}\right)$$

This makes the discrete chain SCFT as fast as the continuous chain SCFT with $O(M \log M)$ complexity per step.

### Memory Modes (GPU)

| Mode | Memory Location | Performance | Use Case |
|------|-----------------|-------------|----------|
| Standard | GPU device | Fast | Default |
| ReduceMemory | Pinned host | 2-4× slower | Large systems |

---

## Solver Hierarchy

**Location**: `src/platforms/cpu/CpuSolver.h`, `src/platforms/cuda/CudaSolver.h`

Implements the numerical methods for advancing propagators.

### Class Diagram

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
        │   ├── pseudo_            - Pseudo<T>* helper
        │   ├── transform_forward()
        │   ├── transform_backward()
        │   └── update_laplacian_operator()
        │       │
        │       ├── CpuSolverPseudoRQM4<T>
        │       │   ├── update_dw(): exp(-w·ds/2)
        │       │   └── advance_propagator(): Richardson
        │       │
        │       └── CpuSolverPseudoDiscrete<T>
        │           ├── update_dw(): exp(-w·ds)
        │           ├── advance_propagator(): bond convolution
        │           └── advance_propagator_half_bond_step()
        │
        └── CpuSolverCNADI<T>                  [Finite Difference]
            └── CN-ADI with tridiagonal solver
```

### Design Notes

- `CpuSolverPseudoBase` consolidates shared FFT logic
- Derived classes override only chain-model-specific methods
- GPU solvers (`CudaSolver*`) follow identical hierarchy with CUDA kernels

---

## FFT Hierarchy

**Location**: `src/platforms/cpu/FFT.h`, `src/platforms/cpu/FftwFFT.h`

Provides spectral transforms supporting all boundary condition types.

### Class Diagram

```
FFT<T>                                        [Abstract Base]
├── forward(T* real, double* spectral)        [pure virtual]
├── backward(double* spectral, T* real)       [pure virtual]
├── forward(T* real, complex* spectral)       [pure virtual, periodic only]
└── backward(complex* spectral, T* real)      [pure virtual, periodic only]
        │
        ├── FftwFFT<T, DIM>                    [CPU/FFTW]
        │   └── DIM = 1, 2, or 3 (compile-time)
        │
        └── CudaFFT<T, DIM>                   [GPU/cuFFT + CudaRealTransform]
            ├── forward_stream(..., cudaStream_t)
            ├── backward_stream(..., cudaStream_t)
            └── Uses CudaRealTransform for non-periodic BC (DCT/DST)
```

### Transform Types

| Boundary | Forward | Backward | Notes |
|----------|---------|----------|-------|
| Periodic | FFT | IFFT | Complex coefficients |
| Reflecting | DCT-II | DCT-III | Real coefficients |
| Absorbing | DST-II | DST-III | Real coefficients |

### Mixed Boundary Conditions

For grids with different BCs per dimension, transforms are applied dimension-by-dimension:

```
3D grid with BC = [periodic, reflecting, absorbing]
  → x: FFT
  → y: DCT-II
  → z: DST-II
```

---

## AndersonMixing Hierarchy

**Location**: `src/common/AndersonMixing.h`

Accelerates SCFT convergence using Anderson mixing algorithm.

### Class Diagram

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

### Algorithm

1. Simple mixing until error < `start_error`
2. Anderson mixing with adaptive parameter
3. History managed via circular buffers

---

## Molecules and Polymer Classes

**Location**: `src/common/Polymer.h`, `src/common/Molecules.h`

Defines polymer architectures and system composition.

### Class Diagram

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

### Polymer Architecture

Polymers are represented as **acyclic graphs** (trees):
- **Vertices**: Junction points and chain ends
- **Edges**: Polymer blocks connecting vertices

```
Example: ABC Triblock (linear)

    0 ----A---- 1 ----B---- 2 ----C---- 3

    blocks = [
        ["A", 0.33, 0, 1],
        ["B", 0.34, 1, 2],
        ["C", 0.33, 2, 3]
    ]

Example: Star Polymer (3-arm)

           1
           |
           A
           |
    2--B-- 0 --C-- 3

    blocks = [
        ["A", 0.33, 0, 1],
        ["B", 0.33, 0, 2],
        ["C", 0.34, 0, 3]
    ]
```

---

## Template Instantiation Strategy

All computational classes use explicit template instantiation for:
- Faster compilation (templates compiled once)
- Smaller binary size
- Better error messages

### Instantiation Macros

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

## Usage Examples

### Basic SCFT Simulation Setup

```python
from polymerfts import SCFT

# Define parameters
params = {
    "nx": [32, 32, 32],
    "lx": [4.0, 4.0, 4.0],
    "chain_model": "continuous",
    "ds": 0.01,
    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 20.0},
    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]]
    }],
    "optimizer": {
        "name": "am",
        "max_hist": 20,
        "start_error": 1e-2,
        "mix_min": 0.1,
        "mix_init": 0.1
    }
}

# Run SCFT
scft = SCFT(params)
scft.run()
```

### Low-Level C++ Usage

```cpp
#include "PlatformSelector.h"

// Create factory (use "cuda" or "cpu-fftw")
AbstractFactory<double>* factory = PlatformSelector::create_factory_real("cuda", false);

// Create computation box
std::vector<int> nx = {32, 32, 32};
std::vector<double> lx = {4.0, 4.0, 4.0};
std::vector<std::string> bc = {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"};
ComputationBox<double>* box = factory->create_computation_box(nx, lx, bc);

// Create molecules
std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
Molecules* molecules = factory->create_molecules_information("continuous", 0.01, bond_lengths);
molecules->add_polymer(1.0, {{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}});

// Create optimizer and solver
PropagatorComputationOptimizer* optimizer = factory->create_propagator_computation_optimizer(molecules, true);
PropagatorComputation<double>* solver = factory->create_propagator_computation(box, molecules, optimizer, "rqm4");

// Compute propagators (w_A, w_B are pre-allocated arrays)
std::map<std::string, double*> w_fields = {{"A", w_A}, {"B", w_B}};
solver->compute_propagators(w_fields);
solver->compute_concentrations();

// Get results
double Q = solver->get_total_partition(0);
double* phi_A = solver->get_total_concentration("A");

// Clean up
delete solver;
delete optimizer;
delete molecules;
delete box;
delete factory;
```

### Using PropagatorSolver (Python)

```python
from polymerfts import PropagatorSolver
import numpy as np

# Create solver
solver = PropagatorSolver(
    nx=[64], lx=[4.0],
    ds=0.01,
    bond_lengths={"A": 1.0},
    bc=["reflecting", "reflecting"],
    chain_model="continuous",
    numerical_method="rqm4",  # or "rk2", "etdrk4", "cn-adi2", "cn-adi4-lr"
    platform="cpu-fftw",
    reduce_memory=False
)

# Add polymer
solver.add_polymer(1.0, [["A", 1.0, 0, 1]])

# Compute propagators
w = np.zeros(64)
solver.compute_propagators({"A": w})

# Get results
Q = solver.get_partition_function(0)
q = solver.get_propagator(polymer=0, v=0, u=1, step=50)
phi_A = solver.get_concentration("A")
```

---

## Design Decisions

### Why Abstract Factory?

- **Platform abstraction**: Same user code works on CPU and GPU
- **Runtime selection**: Platform chosen at initialization, not compile time
- **Encapsulation**: Implementation details hidden from users

### Why Template on Dimension for FFT?

- **Performance**: Compile-time dimension avoids runtime branching
- **Type safety**: Dimension mismatches caught at compile time
- **Code reuse**: Same interface for all dimensions

### Why Graph Representation for Polymers?

- **Generality**: Supports arbitrary branched architectures
- **Efficiency**: Adjacency list enables O(1) neighbor lookup
- **Optimization**: PropagatorCode detects and reuses equivalent computations

### Why Separate Continuous/Discrete Classes?

- **Algorithm differences**: Richardson extrapolation vs bond convolution
- **Memory layout**: Discrete needs half-step propagators
- **Performance**: Specialized implementations avoid runtime checks

---

## References

### Chain Propagator Computation
- D. Yong and J. U. Kim, "Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations," *J. Chem. Theory Comput.* **2025**, 21, 3676.

### Discrete Chain Model
- S. J. Park, D. Yong, Y. Kim, and J. U. Kim, "Numerical implementation of pseudo-spectral method in self-consistent mean field theory for discrete polymer chains," *J. Chem. Phys.* **2019**, 150, 234901.

### Multi-Monomer Polymer Field Theory
- D. Morse, D. Yong, and K. Chen, "Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems," *Macromolecules* **2025**, 58, 816.

### CUDA Implementation
- G. K. Cheong, A. Chawla, D. C. Morse, and K. D. Dorfman, "Open-source code for self-consistent field theory calculations of block polymer phase behavior on graphics processing units," *Eur. Phys. J. E* **2020**, 43, 15.
- D. Yong, Y. Kim, S. Jo, D. Y. Ryu, and J. U. Kim, "Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces," *Macromolecules* **2021**, 54, 11304.

### Langevin FTS
- M. W. Matsen and T. M. Beardsley, "Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation," *Polymers* **2021**, 13, 2437.

### Field Update Algorithms
- B. Vorselaars, "Efficient Langevin and Monte Carlo sampling algorithms: the case of field-theoretic simulations," *J. Chem. Phys.* **2023**, 158, 114117.
- A. Arora, D. C. Morse, F. S. Bates, and K. D. Dorfman, "Accelerating self-consistent field theory of block polymers in a variable unit cell," *J. Chem. Phys.* **2017**, 146, 244902.

### Complex Langevin FTS
- V. Ganesan and G. H. Fredrickson, "Field-theoretic polymer simulations," *Europhys. Lett.* **2001**, 55, 814.
- K. T. Delaney and G. H. Fredrickson, "Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions," *J. Phys. Chem. B* **2016**, 120, 7615.
- J. D. Willis and M. W. Matsen, "Stabilizing complex-Langevin field-theoretic simulations for block copolymer melts," *J. Chem. Phys.* **2024**, 161, 244903.
