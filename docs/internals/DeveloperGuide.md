# Developer Documentation

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

The layers, from the Python simulation classes down to the platform code:

```
Python (src/python)
  SCFT / LFTS / CLFTS ───────────── SymmetricPolymerTheory (χN → auxiliary fields)
        │
        ▼
  PropagatorSolver ──────────────── holds the C++ objects of one simulation
        │   _core module (src/pybind11)
════════╪══════════════════════════════════════════════════════════════════════
C++     ▼
  PlatformSelector ──► AbstractFactory<T> = MklFactory | FftwFactory | CudaFactory
        │ creates
        ├── ComputationBox<T>          CpuComputationBox | CudaComputationBox
        ├── Molecules, PropagatorComputationOptimizer
        ├── PropagatorComputation<T>   ComputationContinuous | ComputationDiscrete
        │      │                       | ComputationReduceMemoryContinuous | ...Discrete
        │      └── owns ComputationBackend<T> = CpuComputationBackend | CudaComputationBackend
        │             └── owns the solver: CpuSolver<T> | CudaSolver<T>
        │                    └── owns the transforms: FFT/DCT/DST, CrysFFT, Pseudo tables
        └── AndersonMixing<T>          CpuAndersonMixing | CudaAndersonMixing[ReduceMemory]
```

The computation classes are written once (`src/common/Computation*.cpp`) and decide **what** is computed and in **which order**. Everything that touches grid data goes through `ComputationBackend`, so the CPU and CUDA code differ only below that line.

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

### One SCFT Iteration Through the Layers

```
SCFT.run (src/python/scft.py)
├─ PropagatorSolver.compute_propagators(w)      → ComputationContinuous::compute_propagators
│   ├─ set_fields: check w and q_init, keep w for solvents, backend.update_dw → solver Boltzmann factors
│   ├─ for each scheduler group, its propagators concurrently (OpenMP thread i uses stream i):
│   │     initial condition: set_unity | copy q_init | junction_product | junction_sum, then apply_mask
│   │     advance_propagator n times → solver: e^{-w ds/2} F⁻¹[ e^{-b²k²ds/6} F[ e^{-w ds/2} q ] ] (+ RQM4 extrapolation)
│   │   synchronize
│   └─ Q = inner_product(q, q†) / V
├─ compute_concentrations, per block: set_zero, add_segment_products (Simpson pairs), normalize_concentration; then solvents
├─ compute_stress, per block: block_stress (segment pairs) → solver segment stress; sum the blocks; lattice derivatives
└─ AndersonMixing.calculate_new_fields(w, w_out - w)
```

Reduce-memory mode runs the same steps one propagator at a time, keeping q only at checkpoints and recomputing the other segments when a sum needs them; the results are the same bit for bit.

---

## 2. Abstract Factory Pattern

For adding observables, Langevin integrators or propagator methods, see
[Extensions.md](Extensions.md) and the runnable
[CustomExtensions.py](../../examples/tools/CustomExtensions.py).

**Location**: `src/common/AbstractFactory.h`

The abstract factory provides a unified interface for creating platform-specific objects without exposing implementation details.

### Class Diagram

```
AbstractFactory<T>                    [Abstract Base]
├── create_computation_box()          [pure virtual]
├── create_molecules_information()    [concrete, platform-independent]
├── create_propagator_computation_optimizer(molecules, aggregate)  [concrete, platform-independent]
├── create_propagator_computation()   [concrete: Computation* with create_backend()]
├── create_anderson_mixing()          [pure virtual]
├── display_info()                    [pure virtual]
└── create_backend()                  [pure virtual, protected]
        │
        ├── FftwFactory<T>             [CPU - FFTW transforms]
        │   └── Creates: CpuComputationBox, CpuAndersonMixing,
        │                CpuComputationBackend (FFTW)
        │
        ├── MklFactory<T>              [CPU - MKL transforms]
        │   └── Creates: CpuComputationBox, CpuAndersonMixing,
        │                CpuComputationBackend (MKL)
        │
        └── CudaFactory<T>            [GPU]
            └── Creates: CudaComputationBox, CudaAndersonMixing[ReduceMemory],
                         CudaComputationBackend
```

`Computation*` is one of the platform-neutral computations (`ComputationContinuous`,
`ComputationDiscrete`, `ComputationReduceMemoryContinuous`,
`ComputationReduceMemoryDiscrete`). `create_propagator_computation()` in
`src/common/AbstractFactory.cpp` resolves and validates the numerical method once,
chooses the computation from the chain model and `reduce_memory`, and builds it around
the backend that the platform returns from `create_backend()`. The backend creates the
solver (`CpuSolver*` or `CudaSolver*`) for that method.

### Platform Implementations

The classes each platform provides (the computation classes above them are shared):

| Role | CPU | CUDA |
|------|-----|------|
| Backend | `CpuComputationBackend` | `CudaComputationBackend` |
| Factory | `MklFactory`, `FftwFactory` | `CudaFactory` |
| Computation box | `CpuComputationBox` | `CudaComputationBox` |
| Solver interface | `CpuSolver<T>` | `CudaSolver<T>` |
| Pseudo-spectral base | `CpuSolverPseudoBase<T>` | `CudaSolverPseudoBase<T>` |
| Continuous, RQM4 | `CpuSolverPseudoRQM4` | `CudaSolverPseudoRQM4` |
| Continuous, RK2 | `CpuSolverPseudoRK2` | `CudaSolverPseudoRK2` |
| Discrete chains | `CpuSolverPseudoDiscrete` | `CudaSolverPseudoDiscrete` |
| Real space, CN-ADI2 | `CpuSolverCNADI` | `CudaSolverCNADI` |
| Transforms | `FftwFFT<T,DIM>`, `MklFFT<T,DIM>` (FFT, DCT, DST) | periodic: cuFFT plans in `CudaSolverPseudoBase`; DCT/DST: `CudaFFT<T,DIM>` over `CudaRealTransform` |
| Fourier-space tables | `Pseudo<T>` | `CudaPseudo<T>` (device copy) |
| CrysFFT (space groups) | `{Fftw,Mkl}CrysFFT{Pmmm,Recursive3m,ObliqueZ}` | `CudaCrysFFT` (Pmmm), `CudaCrysFFTRecursive3m`, `CudaCrysFFTObliqueZ` |
| Field update | `CpuAndersonMixing` | `CudaAndersonMixing`, `CudaAndersonMixingReduceMemory` |

### Platform Selection

```cpp
#include "PlatformSelector.h"

// Create factory for a specific platform ("cpu-fftw", "cpu-mkl", or "cuda")
std::unique_ptr<AbstractFactory<double>> factory =
    PlatformSelector::create_factory_real("cuda", reduce_memory);

// For complex-valued field theories
std::unique_ptr<AbstractFactory<std::complex<double>>> factory_complex =
    PlatformSelector::create_factory_complex("cuda", reduce_memory);
```

**Notes:**

- The only recognized platform strings are `"cpu-fftw"`, `"cpu-mkl"`, and `"cuda"`; any other name (including `"auto"`) throws `Could not find platform '...'`. Automatic platform selection (GPU for 2D/3D, CPU for 1D) is implemented purely in the Python layer, in `PropagatorSolver._auto_platform` (`src/python/propagator_solver.py`); `scft.py`, `lfts.py` and `clfts.py` just pass `platform="auto"` through to `PropagatorSolver`. It is not in C++.
- The Python binding exposes a static `PlatformSelector.create_factory(platform_name, reduce_memory, type)` (with `type` = `"real"` or `"complex"`) that dispatches to `create_factory_real`/`create_factory_complex` (`src/pybind11/polymerfts_core.cpp`).

### Factory Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `create_computation_box(nx, lx, bc, mask=nullptr)` | `std::unique_ptr<ComputationBox<T>>` | Orthogonal simulation grid with boundary conditions; optional mask for impenetrable regions |
| `create_computation_box(nx, lx, bc, angles, mask=nullptr)` | `std::unique_ptr<ComputationBox<T>>` | Grid with lattice angles in degrees for non-orthogonal cells (`angles` empty: orthogonal; `[gamma]` for 2D; `[alpha, beta, gamma]` for 3D); optional mask |
| `create_propagator_computation_optimizer(molecules, aggregate)` | `std::unique_ptr<PropagatorComputationOptimizer>` | Propagator dependency graph with optional aggregation (concrete method of the base class) |
| `create_molecules_information(model, ds, bond_lengths)` | `std::unique_ptr<Molecules>` | Polymer/solvent container (concrete method of the base class) |
| `create_propagator_computation(box, molecules, optimizer, method, space_group=nullptr)` | `std::unique_ptr<PropagatorComputation<T>>` | Propagator solver (method: rqm4, rk2, cn-adi2); optional `SpaceGroup*` enables space-group symmetry (concrete method of the base class; the platform supplies the backend) |
| `create_anderson_mixing(n_var, max_hist, ...)` | `std::unique_ptr<AndersonMixing<T>>` | Field update accelerator |
| `create_fourier_transform(box, bc)` | `std::unique_ptr<FourierTransform<T>>` | Transforms and Fourier multipliers of whole fields (one transform per axis, see Section 3.4) |

The caller owns the created objects. An object created from others keeps raw pointers
to them (the optimizer to its molecules; the computation to its box, molecules,
optimizer and space group), so they must outlive it. Local `unique_ptr`s are destroyed
in reverse order of declaration, so creating the objects in this order destroys each
computation before the objects it points to. The Python bindings keep these arguments
alive with `py::keep_alive`.

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
│   ├── mask_full_      - impenetrable regions on the full grid (optional)
│   └── mask            - the same, reduced, while a space group is set
├── Methods:
│   ├── integral(g)              - ∫g(r)dr
│   ├── inner_product(g, h)      - ∫g(r)h(r)dr
│   ├── multi_inner_product()    - sum of inner products
│   ├── zero_mean(g)             - remove spatial mean
│   ├── set_lx(lx)               - update box lengths [virtual]
│   ├── set_lattice_parameters() - update with angles [virtual]
│   └── set_space_group(sg)      - reduced basis on/off [virtual]
        │
        ├── CpuComputationBox<T>      [CPU Memory]
        │
        └── CudaComputationBox<T>     [GPU Memory]
```

**Space groups.** `set_space_group(sg)` switches the box to the reduced basis of `sg`
(or back to the full grid for `nullptr`): `get_n_basis()`, the orbit counts and
`get_mask()` follow it. It rejects a group built for another grid, reduces the
full-grid mask afresh every time, and changes nothing if a step fails. A propagator
computation sets the box's group (also to none) when it is created and refuses to run
if the box got another group afterwards, so a box can be reused one computation at a
time but not shared by computations with different groups.

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
├── compute_partition_errors()                [pure virtual]
├── check_total_partition(tolerance)
└── update_laplacian_operator()               [pure virtual]
        │
        └── ComputationCommon<T>              [backend, scheduler, φ storage, getters, solvents]
            │
            ├── ComputationContinuous<T>      full storage; RQM4, RK2 or CN-ADI2
            ├── ComputationDiscrete<T>        full storage; bond convolution and half bonds
            │
            └── ComputationReduceMemoryBase<T> [checkpoints and recomputation]
                ├── ComputationReduceMemoryContinuous<T>
                └── ComputationReduceMemoryDiscrete<T>

ComputationBackend<T>                         [Abstract Base, grid operations]
        ├── CpuComputationBackend<T>          host arrays, OpenMP threads, CpuSolver
        └── CudaComputationBackend<T>         device arrays, CUDA streams, CudaSolver,
                                              checkpoints in pinned host memory
```

All of these live in `src/common/` except the two backends.
`AbstractFactory::create_propagator_computation()` creates the computation with
the backend that the platform factory (`MklFactory`, `FftwFactory`,
`CudaFactory`) supplies; the Python layer and the C++ tests (through
`tests/TestComputationFactory.h`) obtain every computation from a factory.

**What has been computed.** The base class records whether the propagators,
the concentrations and the stress belong to the current fields and box. A new
`compute_propagators()` and `update_laplacian_operator()` clear all three; each
computation sets its own when it completes. The getters throw instead of
returning values of other fields or another box: `get_total_partition()`,
`get_chain_propagator()`, `compute_concentrations()`, `compute_stress()` and
`compute_partition_errors()` need the propagators, the concentration getters
also `compute_concentrations()`, `get_stress()` also `compute_stress()`. A new
computation class sets the flags at the end of its `compute_*` methods and
calls the `require_*` checks first in its getters. `compute_partition_errors()`
returns the forward/backward error of every polymer and prints nothing;
`check_total_partition()` compares them with a tolerance. The molecules are
fixed once a computation exists (`add_polymer()` and `add_solvent()` throw), and
a `cn-adi2` computation keeps its box orthogonal while it exists
(`ComputationBox::hold_orthogonal_cell()`).

**Who does what.** The computation classes decide *what* is computed and in
*which order*: the scheduler's groups of propagators, the initial condition of
each propagator (free end, grafting point, junction product, aggregated sum),
which segment pairs enter the concentration and stress integrals and with
which weights, and the normalization. Their loops read top to bottom in the
order the work happens. Every operation on grid data goes through
`ComputationBackend` (`src/common/ComputationBackend.h`): advance a
propagator, build a junction, accumulate segment products, normalize a block,
sum the stress of a list of segment pairs, inner products, checkpoints.

**Rules for the backend interface.**

- One operation is one physically meaningful step on whole fields. Do not add
  array-level helpers (scale, add, multiply): they force one kernel launch per
  call and keep a backend from fusing the work of a step.
- Fields are `BackendArray<T>*` (host memory on the CPU, device memory on
  CUDA); checkpoints are `CheckpointArray<T>*` (pinned host memory on CUDA)
  and may only be stored, loaded, or used as junction sources. Both types are
  incomplete, so common code cannot dereference them.
- An operation with a `stream` argument may run concurrently with the other
  streams (an OpenMP thread on the CPU, a CUDA stream on the GPU); call
  `synchronize()` before using results from several streams.
- A new feature goes into the computation class once, plus the backend
  operations it needs on both platforms. The CPU implementation is the plain
  loop and serves as the reference; the CUDA one may fuse, batch or overlap as
  long as `tests/` and the CPU-CUDA equivalence tests pass.

**Reduce-memory mode.** The checkpointed classes store q(r, n) at the start of
every propagator, at the segments other propagators start from, and every
⌈2√(total segments)⌉ segments, and recompute the other segments from the last
checkpoint before them. The solvers are deterministic and the classes use the
same segment pairs, weights and summation order as full storage, so both
modes give bit-for-bit identical results. Propagators are computed one at a
time on one stream.

**Chain Models:**

| Model | Propagator Equation | Solver Method |
|-------|---------------------|---------------|
| Continuous | $`\partial q/\partial s = (b^2/6)\nabla^2 q - wq`$ | RQM4, RK2, or CN-ADI2 |
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
        │   ├── fft_               - std::unique_ptr<FFT<T>> for transforms
        │   ├── transform_forward()
        │   ├── transform_backward()
        │   ├── update_laplacian_operator()
        │   ├── compute_single_segment_stress()  [shared implementation]
        │   └── get_stress_boltz_bond()          [virtual; base returns nullptr]
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
        │           ├── advance_propagator_half_bond_step()
        │           └── get_stress_boltz_bond(): returns boltz_bond
        │
        └── CpuSolverCNADI                     [Finite Difference; CpuSolver<double>, real fields only]
            └── CN-ADI with tridiagonal solver
```

The CUDA side mirrors this layout. `CudaSolver<T>` is the abstract base;
`CudaSolverPseudoBase<T>` (`src/platforms/cuda/CudaSolverPseudoBase.h`) holds what the
three pseudo-spectral solvers share -- it uses the computation box and the backend's streams,
and owns the `Pseudo` helper, the stress reduction workspace, the space-group work buffers
and the CrysFFT fast path --
and implements `set_space_group()`, `update_laplacian_operator()` and the shared
`update_dw_impl()`. It also owns the spectral transforms: cuFFT plans for periodic boundaries
(`fft_forward()` / `fft_backward()`, batch of 1 or 2 fields, the largest batch is a
constructor argument) and per-stream DCT/DST objects (`fft_`) for non-periodic ones.
The physical-grid ↔ reduced-basis maps of the CrysFFT fast path are built by the
platform-independent `CrysFFTBasisMapping.h` and shared with the CPU solver.
It also implements `compute_single_segment_stress()` once for the three solvers
(as `CpuSolverPseudoBase` does on the CPU): the derived classes only lend their
work buffers through `stress_scratch()`, and the discrete solver overrides
`get_stress_boltz_bond()` (the base default, nullptr, is the continuous chain).
`CudaSolverPseudoRQM4`, `CudaSolverPseudoRK2` and `CudaSolverPseudoDiscrete` keep
their work buffers, `update_dw()`/`advance_propagator()` and the stress scratch they lend;
`CudaSolverCNADI` is the finite-difference solver. The non-periodic DCT/DST
objects are held through `CudaFFTBase<T>`, whose `forward_stream()` /
`backward_stream()` run the transform on the propagator's stream. Mechanical real/complex
differences (D2Z vs Z2Z transforms, CUB sum vs reduce) are hidden behind the overloads
in `CudaTypeDispatch.h`, so a remaining `if constexpr` on the field type marks a genuine
algorithmic difference or a real-only feature and is commented as such.

### 3.4 FFT Hierarchy

**Location**: `src/platforms/cpu/FFT.h`, `src/platforms/cpu/FftwFFT.h`, `src/platforms/cpu/MklFFT.h`, `src/platforms/cuda/CudaFFT.h`

```
FFT<T>                                        [Abstract Base]
├── forward(T* real, double* spectral)        [pure virtual]
├── backward(double* spectral, T* real)       [pure virtual]
├── forward(T* real, complex* spectral)       [pure virtual, periodic only]
└── backward(complex* spectral, T* real)      [pure virtual, periodic only]
        │
        ├── FftwFFT<T, DIM>                    [CPU - FFTW backend]
        │   └── DIM = 1, 2, or 3 (compile-time)
        │
        ├── MklFFT<T, DIM>                     [CPU - MKL backend]
        │   └── DIM = 1, 2, or 3 (compile-time)
        │
        └── CudaFFTBase<T>                    [GPU abstract base]
            ├── forward_stream(..., cudaStream_t)   [pure virtual; double* and complex* overloads]
            ├── backward_stream(..., cudaStream_t)  [pure virtual; double* and complex* overloads]
            │
            └── CudaFFT<T, DIM>               [GPU/cuFFT + CudaRealTransform]
                ├── forward()/backward() call the stream versions on stream 0
                └── Uses CudaRealTransform for non-periodic BC (DCT/DST)
```

**Transform Types:**

| Boundary | Forward | Backward | Notes |
|----------|---------|----------|-------|
| Periodic | FFT | IFFT | Complex coefficients |
| Reflecting | DCT-II | DCT-III | Real coefficients |
| Absorbing | DST-II | DST-III | Real coefficients |

**Fourier transform of whole fields** (`src/common/FourierTransform.h`,
`CpuFourierTransform`, `CudaFourierTransform`): the Python layer reaches the platform
FFT through `PropagatorSolver.fourier`. `FFT<T>` transforms one field in the memory of the
platform; `FourierTransform<T>` works on host arrays of several fields and keeps multipliers. It transforms host fields (`forward`, `backward`)
and evaluates `backward(Σ M(k) · forward(fields))` for Fourier multipliers stored once
(`create_multiplier`, held by a `FourierMultiplier<T>` handle): matrices of
`n_out × n_in` functions of k, or one function applied to each field. The CUDA version
keeps the multipliers in device memory and runs a call on its own stream: upload,
transforms, a grid-stride product kernel, inverse transforms and download. The
coefficients have the layout of `FFT<T>`, and `get_k_squared()` gives |k|² in it for the
current box with the alias convention of `Pseudo`. For real fields the multipliers and the
coefficients given to `backward` are made Hermitian on the planes of the half spectrum that
hold both k and −k, where the inverse r2c transforms of FFTW, MKL and cuFFT would read
other values differently; on the device every field starts an even number of doubles from
the next (cuFFT needs 16-byte aligned real data). Periodic and wall axes cannot be mixed;
`PropagatorSolver.fourier` uses the cosine series along walls, whose fields have no
boundary value. The Langevin steps use it for smearing (except in boxes that mix periodic
and wall axes), the transforms of ψ, the `etd` integrator, the dilation of the smearing
(osmotic pressure), the LR compressor of L-FTS in periodic boxes and in boxes with reflecting
walls only (boxes that mix periodic and wall axes use `scipy.fft`; absorbing walls are refused)
and, in periodic boxes, WTMD and the samples for the structure functions; boxes with walls
record the Fourier coefficients of `numpy.fft`.

### 3.5 AndersonMixing Hierarchy

**Location**: `src/common/AndersonMixing.h`

```
AndersonMixing<T>                             [Abstract Base]
├── n_var           - number of field variables
├── max_hist        - maximum history length
├── mix_min/mix_init - mixing parameter bounds
├── calculate_new_fields(w_new, w_current, w_deriv, old_error_level, error_level) [pure virtual]
└── reset_count()   [virtual, empty default; overridden by the platform classes]
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
├── contour_length  - length relative to N_Ref (may exceed 1)
├── v, u            - connecting vertices
└── n_segment       - discretized segment count

Polymer                                       [Single Chain Type]
├── alpha           - total contour length
├── volume_fraction - system fraction
├── blocks[]        - vector of Block
├── adjacent_nodes  - graph adjacency list
└── Methods:
    ├── get_block(v, u), get_n_blocks()
    ├── get_adjacent_nodes()
    └── display_architecture()

Molecules                                     [System Container]
├── model_name      - "continuous" or "discrete"
├── ds              - contour step size
├── bond_lengths    - {monomer: a/a_ref}
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
INSTANTIATE_CLASS(ComputationContinuous);
// Expands to:
//   template class ComputationContinuous<double>;
//   template class ComputationContinuous<std::complex<double>>;

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
cmake --build build --parallel 4
```

CTest uses the staged package in `build/python`. To update the installed package,
build and install a wheel as described in
[Installation](../getting-started/Installation.md).

#### Python Changes

Changes to `src/python/*.py` are picked up by the tests after
`cmake --build build --target polymerfts_stage` copies them into
`build/python/polymerfts`. Rebuild and reinstall the wheel to update the package
used by examples and tutorials.

#### Running Tests

See [Installation.md](../getting-started/Installation.md#testing) for test commands.

---

## 6. Adding New Features

### Platform Implementation

When adding new computational features, you must implement them for both platforms unless explicitly platform-specific. Where each kind of change goes:

| To change | Edit |
|-----------|------|
| Order of a computation, which segments enter a sum, normalization | `src/common/Computation*.cpp` (once, for both platforms) |
| A new kind of grid operation | `src/common/ComputationBackend.h` and both backends |
| A numerical method (advance step, segment stress) | the solvers in `src/platforms/cpu` and `src/platforms/cuda` |
| Transforms (FFT, DCT, DST) and CrysFFT | the FFT and CrysFFT classes ([FFTImplementation.md](FFTImplementation.md)) |
| Fourier-space tables | `src/common/Pseudo.cpp` (`CudaPseudo` uploads them) |
| Which propagators are computed, aggregation | `PropagatorComputationOptimizer`, `PropagatorAggregator` |
| Concurrent groups | `Scheduler` |
| Which objects a platform creates | the factories |
| Simulation loop and parameters | `src/python` |

A change to how propagators, concentrations or stress are assembled belongs in
the computation classes in `src/common/` (Section 3.2); a new kind of grid
operation is a new `ComputationBackend` operation implemented by both backends.

### Memory Management

Who owns whom:

- `PropagatorSolver` (Python) holds the box, the molecules, the optimizer and the computation; the bindings keep the arguments of each object alive while it exists.
- A computation owns its backend, the scheduler, the concentration fields and the propagator storage; the backend is deleted last.
- A backend owns its solver, and on CUDA the streams, events and scratch buffers.
- A solver owns its transforms, its `Pseudo` tables, its CrysFFT objects and its Boltzmann factors.

Allocations are owned by objects that release them, so destructors and constructors
that throw need no cleanup code:

- device memory: `CudaDeviceBuffer<T>`; page-locked host memory: `CudaPinnedBuffer<T>`
  (both in `src/platforms/cuda/CudaCommon.h`; move-only, `allocate(n)`, `release()`,
  `get()`, `empty()`, and an implicit conversion that lends the raw pointer to kernels,
  `cudaMemcpy` and cuFFT; template calls such as CUB take `get()`, and tests for null use
  `empty()`, because the second conversion to `void*`, with which GCC and Clang reject
  `delete buffer`, makes `if (buffer)` ambiguous too); a failed allocation throws with its size and
  the caller's file, line and function (`std::source_location`);
- host arrays: `std::vector`; owned objects (solvers, transforms, CrysFFT engines,
  `Pseudo`, the scheduler, the backend): `std::unique_ptr`.

Raw pointers are lent, never owned. The factory functions (`create_computation_box`,
`create_propagator_computation`, ...) return `std::unique_ptr`, which the Python bindings
take over as the holders of their objects. Two places hand out ownership through raw
pointers on purpose: the backends' `allocate()`/`deallocate()` and
`allocate_checkpoint()`/`deallocate_checkpoint()` pairs,
the allocators of the propagator handles and the reduce-memory checkpoints, which the
computation classes free in one place; and the `CudaCommon` singleton, which is created
with `new` and deliberately never deleted (it lives until the process exits). cuFFT plans, CUDA streams and events are not memory and are destroyed
explicitly where their order matters (the solver before the streams it runs on); they are
created through `count_cufft_plan(...)`, `count_cuda_stream(...)` and `count_cuda_event(...)`
and destroyed with `release_cufft_plan()`, `release_cuda_stream()` and
`release_cuda_event()`, which keep live counts (`cuda_handle_counters()`); every release
helper, like `counted_cuda_release()`, takes the resource off the count even when the
release call fails, and counts the failure in `cuda_release_failures()`. Events used
only for profiling are `CudaEvent` objects. The CPU FFT classes keep their FFTW/MKL
plans and descriptors and the `fftw_malloc`-aligned buffers of `FftwFFT`/`FftwCrysFFT*`
in hand-written `release()` functions, which their destructors call.

Every owned CUDA allocation (the buffers and the backends' propagator and checkpoint
arrays) goes through `counted_cuda_allocate()`/`counted_cuda_release()`, which keep exact
counts of the live device and pinned allocations (`CudaDeviceMemory::counters()`,
`CudaPinnedMemory::counters()`); `cuda_buffer_fail_countdown()` is a test hook that makes
the N-th following counted allocation throw. `cudaFree` and `cudaFreeHost` go through
`ignore_release_error()`, which does not throw, clears the error a failed release leaves in
the per-thread slot and counts it in `cuda_release_failures()` (the plan, stream and event
release helpers count their failures there too; cuFFT has its own runtime and leaves
nothing in the slot). `tests/TestCudaDeviceBuffer.cu` uses all of them: every kind of CUDA
object (all solvers, the CrysFFT engines, non-periodic transforms, complex fields, the
Anderson mixers), built with 4 streams, must return every counter (allocations, plans,
streams, events) to its start when destroyed without a failed release, and a sweep with 2
streams fails each counted allocation of each construction in turn and requires the same
after every failure (failures of cuFFT plan, stream and event creation are not injected).
The test also lets a small construction run out of device memory for real, by holding the
rest of the free memory in a ballast buffer for a moment, so it runs serially; when the
memory cannot be held it reports a ctest skip. Freeing a buffer by hand does not compile
(`cudaFree`, `cudaFreeHost` and `cudaFreeAsync` have deleted overloads for buffers). GCC
and Clang also reject `delete buffer` as ambiguous; the standard does not require that
(only conversions to object pointer types count for `delete`), so a stricter compiler
may accept it.

### Propagator Computation Optimizer

The `PropagatorComputationOptimizer` automatically detects redundant calculations using hash tables of `PropagatorCode` objects. Avoid manual optimization.

### New Monomer Types or Interactions

Modify only the parameter dictionary - the code supports arbitrary numbers of monomer types. The `SymmetricPolymerTheory` class handles interaction matrix eigendecomposition automatically.

### New Numerical Methods

1. Implement the solver in both `cpu/` and `cuda/` directories
2. Add its name to the method check in the factory classes and the solver choice to the constructors of both backends (`CpuComputationBackend`, `CudaComputationBackend`)
3. Update parameter validation in Python modules
4. Add tests and benchmarks

---

## 7. Code Standards

### C++

- Use C++20 features where appropriate
- Follow existing naming conventions
- Document public APIs
- CUDA error checks follow the CUDA C++ Best Practices Guide (2.3): wrap every call that returns
  an error code (`CUDA_CHECK(...)` for runtime and CUB calls, `CUFFT_CHECK(...)` for
  cuFFT) and follow every kernel launch with `CUDA_CHECK_LAUNCH();`. The check
  reads a host-side value and does not synchronize; `cudaGetLastError()` clears the slot, so an
  error is reported at the launch that caused it. Release calls run in destructors and
  `release()` paths and do not throw: wrap `cudaFree`/`cudaFreeHost` in
  `ignore_release_error(...)`, so a failed one is not reported later by an unrelated check,
  and destroy plans, streams and events with the counting `release_*` helpers.

### Python

- Follow PEP 8 style guidelines
- Add docstrings for public functions
- Include type hints where practical

---

## 8. Debugging Tips

### Segmentation Faults

See [Installation.md](../getting-started/Installation.md#segmentation-fault) for stack size settings.

### CUDA Debugging

- A launch error (bad configuration, too much shared memory) is reported by the check right
  after its launch.
- An error inside a running kernel (e.g. an illegal address) is not seen by its launch check.
  It is reported by the first CUDA call made after the fault, at the latest by the next call that
  waits for the GPU, and that call names its own line. Run with `CUDA_LAUNCH_BLOCKING=1` to make
  every launch synchronous, so the error appears at its own launch check (slower; for debugging
  only).
- An allocation that fails names its size and the code that asked for it, e.g.
  `allocation of 268435456 bytes of device memory; File: ".../CudaPseudo.cu", line: ...,
  function <...>: out of memory`.
- `compute-sanitizer --tool memcheck` finds out-of-bounds accesses and, with
  `--report-api-errors all`, every CUDA call that returns an error.
- Use Nsight Systems / Nsight Compute for profiling.

### Platform Consistency Issues

If GPU and CPU give different results:
1. Check floating-point operation order
2. Verify FFT normalization
3. Check memory alignment

### Verifying Numerical Accuracy

1. Run same test on both `cuda` and CPU platforms - results should match to ~10⁻¹³
2. Check partition function consistency: `solver.check_total_partition()` should pass
   (`solver.partition_errors()` returns the relative error of each polymer)
3. Verify material conservation: `np.mean(sum(phi_species))` = 1.0 within round-off (10⁻¹⁵–10⁻¹⁴)

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
