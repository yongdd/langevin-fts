# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Note**: Mathematical equations use LaTeX syntax (`$...$` for inline, `$$...$$` for display). These render correctly on GitHub and in editors with LaTeX support.

## Table of Contents

- [Project Overview](#project-overview)
- [Critical Rules](#critical-rules)
- [Requirements](#requirements)
- [Build System](#build-system)
- [Testing](#testing)
- [Code Architecture](#code-architecture)
- [Running Simulations](#running-simulations)
- [Troubleshooting](#troubleshooting)
- [Chain Models](#chain-models)
- [Numerical Methods](#numerical-methods)
- [Units and Conventions](#units-and-conventions)
- [Development Notes](#development-notes)
- [Documentation and Learning](#documentation-and-learning)
- [Key References](#key-references)

## Project Overview

This is a polymer field theory simulation library implementing Self-Consistent Field Theory (SCFT) and Langevin Field-Theoretic Simulations (L-FTS). The codebase consists of high-performance C++/CUDA computational kernels exposed through Python interfaces via pybind11, enabling efficient polymer physics simulations with Python's ecosystem.

## Critical Rules

> **These rules are mandatory and must be followed at all times.**

### Never Commit Without Permission

Always wait for explicit user approval before running `git commit`. The user must explicitly say "commit" or "make a commit" - do NOT interpret "update", "add", "change", or "fix" as permission to commit. After making changes, ask "Should I commit this change?" and wait for confirmation.

### Never Modify Test Parameters

Test files in `tests/` contain carefully calibrated parameters. The following modifications are **strictly forbidden** unless the user explicitly requests them:

- NEVER increase tolerance values (e.g., changing 1e-7 to 1e-6 to make a test pass)
- NEVER decrease field strength or standard deviation values
- NEVER change grid sizes, box dimensions, or polymer parameters
- NEVER weaken any test conditions to make tests pass

If a test fails, **report the failure to the user** rather than modifying the test to pass. The test parameters are designed to catch real bugs - weakening them hides problems instead of fixing them.

### Leave `devel/` Alone

`devel/` is the maintainer's private working folder. Do not audit, clean up, reformat or commit anything in it, and never link to it from the documentation. Its files frequently appear as modified in `git status`; that is expected.

### Use SLURM for Long-Running Jobs

For any computation expected to take longer than 1 minute, submit it as a SLURM job instead of running directly. This includes benchmarks, SCFT convergence tests, and parameter sweeps. Use `sbatch` to submit jobs and launch multiple jobs simultaneously when running parameter studies or benchmarks with different configurations. Example:

```bash
# Submit multiple jobs in parallel
for param in 100 200 400 800; do
    sbatch --job-name="test_$param" --wrap="python script.py --param $param"
done
```

### Split Large Computations

For every computation, launch a separate SLURM job for each parameter value instead of queuing all parameters in a single job. This allows parallel execution and avoids blocking on slow computations.

## Requirements

### Software Dependencies

| Dependency | Minimum Version | Notes |
|------------|-----------------|-------|
| spglib | - | Space group symmetry (required) |
| LAPACKE | - | Anderson mixing SVD solver (required) |
| CMake | 3.17+ | Build system (3.25.2+ for the GPU backend: C++20 for nvcc) |
| C++ Compiler | C++20 support | GCC 10+, Clang 10+ (GPU backend: GCC 11+, Clang 15+ for `std::source_location`) |
| Python | 3.11+ | With NumPy 2.0+, SciPy 1.14+ |
| CUDA Toolkit | 12.2+ | For GPU support (device code is C++20) |
| MKL | - | CPU FFT backend (via conda) |
| FFTW | 3.3+ | Alternative CPU backend (GPL, optional) |
| pybind11 | 2.13+ | Python bindings |

### Conda Environment

```bash
# Activate the conda environment
conda activate polymerfts
```

## Build System

### Build Commands

```bash
# Initial build (from repository root)
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_MKL=ON -DPOLYMERFTS_USE_FFTW=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make -j8

# Install Python modules to conda environment (for examples/tutorials; tests use the build tree)
make install

# Run basic tests (installation verification, ~1 min)
ctest -L basic

# Run full tests (development validation, ~7 min)
ctest

# Clean build (if needed)
cd build && rm -rf * && cmake ../ -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_MKL=ON -DPOLYMERFTS_USE_FFTW=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX && make -j8 && make install
```

### Important Build Notes

- Requires C++20 standard for host and device code (set in CMakeLists.txt)
- CUDA Toolkit 12.2+ required for GPU support: nvcc 11.x has no `-std=c++20`, and nvcc 12.0/12.1 cannot evaluate `std::source_location::current()`; configuring with an older toolkit, CMake older than 3.25.2, or a host compiler without `std::source_location` stops with an error. The cuFFT library is taken from the toolkit of that nvcc, not from a conda environment's CUDA libraries
- Compute capabilities 60-90 are compiled by default; pass `-DPOLYMERFTS_CUDA_ARCHITECTURES="80;86"` to compile for other GPUs
- If encountering "Unsupported gpu architecture" errors, drop the unsupported capabilities from that list
- Debug builds: Change `CMAKE_BUILD_TYPE` to `Debug` for additional warnings and profiling symbols
- `-DPOLYMERFTS_WERROR=ON` turns host C++ warnings into errors (CI builds with it); keep new C++ warning-free under GCC 13
- The build links libraries by absolute path and passes the CUDA directories explicitly; it does not depend on `CPLUS_INCLUDE_PATH` or `LIBRARY_PATH` of the shell. Configure with `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX`
- **User builds**: CMakeLists.txt defaults are both OFF. Users who only need one backend can build without these flags.

### Environment Setup

```bash
# Activate conda environment (required before any work)
conda activate polymerfts

# Set stack size (important to avoid segfaults)
ulimit -s unlimited
export OMP_STACKSIZE=1G
```

### GPU Selection (Important!)

**At the start of each session**, run `nvidia-smi` to check GPU utilization and select an idle GPU:

```bash
# Check GPU status
nvidia-smi

# Set CUDA_VISIBLE_DEVICES to an idle GPU before running tests
export CUDA_VISIBLE_DEVICES=0  # or 1, 2, etc. based on which GPU is idle
```

This prevents conflicts with other users and ensures consistent test results.

## Testing

### Basic vs Full Tests

Two test modes are available:

| Mode | Command | Tests | Time | Purpose |
|------|---------|-------|------|---------|
| **Basic** | `ctest -L basic` | 57 | ~1 min | Installation verification |
| **Full** | `ctest` | 94 | ~7 min | Development validation |

### Running Tests

**Note:** The Python tests import the package staged in `build/python/` by `make` (target `polymerfts_stage`), so they always test the current build; `make install` is only needed to run examples and tutorials from the installed package.

```bash
# From build directory
cd build

# Basic tests (for users, installation verification)
ctest -L basic

# Full tests (for development)
ctest

# Run specific test (example)
./tests/TestPseudoBranchedContinuous3D
./tests/TestScft3D

# Run tests with verbose output
ctest -V

# Run specific test pattern
ctest -R Pseudo  # runs all tests matching "Pseudo"
```

### Performance and Accuracy Benchmarks

Benchmark scripts are available in `scripts/`:
- `scripts/benchmark_crysfft_triplet.py`: CrysFFT speedup benchmark
- `scripts/benchmark_space_group_speed.py`: Space group performance benchmark

**Running benchmarks with SLURM job scheduler:**

Use the provided SLURM template `scripts/slurm_run.sh` (adjust the Python script as needed):

```bash
sbatch scripts/slurm_run.sh
```

**What the scripts measure:**
- `benchmark_crysfft_triplet.py`: times only the propagator computation of a single AB diblock in a BCC (Im-3m) field, once with the space group on (CrysFFT physical basis auto-selected) and once off, and reports the per-iteration time of both runs (`iter_sec`), the `speedup` and `percent` (CPU-FFTW only; the platform is hard-coded). Options: `--nx`, `--lx`, `--ds`, `--fa`, `--iters`, `--warmup`, `--fields`.
- `benchmark_space_group_speed.py`: runs a fixed number of SCFT iterations for one phase (`--phase BCC|DG|Sigma|HCP`) on one platform (`--platform cpu-fftw|cuda`) with `--space-group on|off` and reports `elapsed_sec`, `iter_sec` and the final `Q`; submit one SLURM job per combination.
- Both print a JSON summary to stdout and write the same JSON to the file given by `--output` (nothing is written when `--output` is omitted).

**Key metrics to verify:**
- Space-group on/off speedup (`speedup` in the triplet JSON; ratio of `iter_sec` between matching speed-benchmark runs). Reference timings are tabulated in `docs/internals/FFTImplementation.md` (Performance Benchmarks).
- Convergence orders are not measured by these scripts; they are guarded by `tests/TestConvergenceOrderLite.py` (RQM4 order > 3.0, 1.5 < RK2 order < 2.5).

### Testing Guidelines for Numerical Methods

When testing propagator solvers or numerical methods, always follow these requirements:

1. **Verify material conservation**: Check that total monomer concentration is conserved (sum of all species concentrations equals 1.0). This catches normalization errors in propagator computations.

2. **Call `check_total_partition()`**: Always verify the total partition function during tests. This validates that forward and backward propagators are consistent and the chain statistics are computed correctly.

3. **Use field amplitudes with std ≈ 5**: When initializing test fields, choose random fields with standard deviation around 5. This provides sufficient field strength to expose numerical errors while remaining in a physically relevant regime. Weak fields (std < 1) may not reveal accuracy issues.

4. **Final validation with Gyroid SCFT**: Run `examples/scft/Gyroid.py` with the **continuous chain model** as the final integration test (should complete within 5.7 seconds on GPU). This complex 3D morphology is sensitive to numerical errors and validates the complete simulation pipeline.

Example test setup:

```python
# Generate test fields with std ~ 5
w = np.random.normal(0.0, 5.0, size=nx)

# After computing propagators, verify:
# 1. Material conservation
total_phi = sum(phi_species)
np.mean(total_phi)  # Should be ~1.0

# 2. Partition function consistency
solver.check_total_partition()  # Should pass without error
```

## Code Architecture

### Platform Abstraction (Abstract Factory Pattern)

The codebase uses the **abstract factory pattern** to support multiple computational platforms (CPU with MKL/FFTW, CUDA GPUs). This is a key architectural feature:

- **Factory Selection**: `src/common/PlatformSelector.cpp` determines available platforms and creates appropriate factory instances
- **Platform Implementations**:
  - `src/platforms/cpu/`: CPU implementations using MKL or FFTW for FFT and linear algebra
  - `src/platforms/cuda/`: CUDA GPU implementations using cuFFT and custom CUDA kernels
- **Common Interfaces**: `src/common/AbstractFactory.h` defines abstract interfaces that all platform implementations must satisfy
- **Computation and backend**: the propagator computations (`src/common/Computation*.cpp`) are written once; every grid operation goes through `ComputationBackend` (`CpuComputationBackend`, `CudaComputationBackend`), which owns the platform's solver

When adding new computational features, you must implement them for both platforms unless explicitly platform-specific: the order of a computation goes into `src/common/Computation*.cpp` once, new grid operations into both backends, numerical methods into both solver families. `docs/internals/Architecture.md` shows the whole class structure on one page.

### Core Components

#### Propagator Computation (`src/common/PropagatorComputation.h`)

The central computational engine. Computes chain propagators using dynamic programming to avoid redundant calculations for branched polymers. The optimization strategy is described in *J. Chem. Theory Comput.* **2025**, 21, 3676.

Key concepts:
- **Chain propagators**: Solutions to modified diffusion equations (continuous) or recursive integral equations (discrete) representing polymer chain statistics
- **Continuous chains**: Pseudo-spectral method with RQM4 or RK2 solving the modified diffusion equation
- **Discrete chains**: Pseudo-spectral method using bond convolution based on Chapman-Kolmogorov equations (N-1 bond model from Park et al. 2019)
- **Real-space method**: CN-ADI (Crank-Nicolson ADI) finite difference solver (beta feature, continuous chains only). CN-ADI2 (2nd-order).
- **Numerical method selection**: Runtime selection via `numerical_method` parameter: `"rqm4"` (RQM4), `"rk2"` (RK2), or `"cn-adi2"` (CN-ADI2)
- **Aggregation**: Automatic detection and reuse of equivalent propagator computations in branched/mixed polymer systems

#### Computation Box (`src/common/ComputationBox.h`)

Manages simulation grid, FFT operations, and boundary conditions. Handles 1D/2D/3D simulations with periodic, reflecting, or absorbing boundary conditions. All numerical methods support all boundary condition types.

#### Polymer and Molecules (`src/common/Polymer.h`, `src/common/Molecules.h`)

Define polymer chain architectures:
- Supports arbitrary acyclic branched polymers (star, comb, dendritic, bottle-brush, etc.)
- Handles mixtures of block copolymers, homopolymers, and random copolymers
- Stores chain topology as directed acyclic graph for propagator computation optimization

#### Anderson Mixing (`src/common/AndersonMixing.h`)

Iterative solver for SCFT equations. Accelerates convergence by mixing field history. Critical for SCFT convergence; parameters in examples show tuned values.

### Python Layer

#### High-Level Simulation Classes

- `src/python/scft.py`: SCFT simulation orchestrator
  - Handles field initialization, iteration loop, convergence checking
  - Anderson mixing optimizer
  - Implements stress calculations for box size optimization
  - Space group symmetry constraints (beta)

- `src/python/lfts.py`: L-FTS simulation orchestrator
  - Implements Langevin dynamics with Leimkuhler-Matthews method
  - Structure function calculations
  - Field compressors (Anderson Mixing, Linear Response)
  - PCG64 random number generator

- **Python layer rule**: researchers, not developers, edit `src/python`. Keep it script-like: each simulation class reads its options straight from the `params` dictionary in its constructor (defaults written where they are used), there are few files (one topic per file: `scft.py`, `lfts.py` with its field compressors, `clfts.py`, `propagator_solver.py`, `polymer_field_theory.py`, `smearing.py`, `wtmd.py`, `utils.py`, `validation.py`), and no dataclass/spec layers. Duplication between `scft.py`, `lfts.py` and `clfts.py` is acceptable when it keeps each file self-contained; shared mechanics (chi N parsing, block processing, random copolymers, parameter-file loading, VTK output) live in `utils.py`.

- `src/python/polymer_field_theory.py`: Multi-monomer field theory transformations
  - Converts between monomer potential fields and auxiliary fields
  - Eigenvalue decomposition of interaction matrices
  - Hamiltonian coefficient calculations for arbitrary χN parameters

#### Python-C++ Binding (`src/pybind11/polymerfts_core.cpp`)

Exposes C++ classes to Python as the `_core` module. All C++ computational objects are accessible through this interface.

### CUDA Implementation Details

CUDA code uses:
- **Multiple streams** (up to 4): Parallel propagator computations on GPU. One OpenMP thread drives each stream, so the stream count is `OMP_NUM_THREADS` (at most 4), and `OMP_MAX_ACTIVE_LEVELS=0` serializes them (on an A10 an iteration then took 1.5x as long at 32³ and 6-9% longer at 64³-128³ than with `1`)
- **cuFFT**: GPU-accelerated Fast Fourier Transforms
- **Shared memory**: Tridiagonal solvers for real-space methods use shared memory for performance
- **Pinned memory**: Host-device transfers use pinned circular buffers for efficiency
- **Kernels loop over their index range**: every element-wise kernel uses a grid-stride loop (`while (i < M) { ...; i += blockDim.x * gridDim.x; }`), so it is correct for any launch configuration; kernels that give one block a row or a whole 1D transform loop over rows with `gridDim.x` and over the row with `blockDim.x`. Write new kernels the same way. The default launch is 1024 blocks of 256 threads (`LFTS_GPU_NUM_BLOCKS`, `LFTS_GPU_NUM_THREADS` override it); line solvers (CN-ADI) size their launch to the number of grid lines
- **No new OpenMP regions in per-iteration host code of the CUDA path**: a `#pragma omp parallel for` in the box-change table update made the Gyroid example 3x slower (its threads competed with spinning threads elsewhere in the process; `OMP_WAIT_POLICY=PASSIVE` hid it) and was removed. Host work there stays serial or moves to the GPU
- **Launch shape**: element kernels launched to cover an array use `grid_blocks(n, block_size)` (`CudaCommon.h`), which caps the grid at `MAX_GRID_THREADS` = 2^20 threads so the stride loop cannot overflow `int`
- **Error checks (CUDA C++ Best Practices Guide 2.3)**: every call that returns an error code is wrapped (`CUDA_CHECK(...)` for runtime and CUB calls, `CUFFT_CHECK(...)` for cuFFT), and every kernel launch is followed by `CUDA_CHECK_LAUNCH();` (one check after an unbraced `if/else` launch pair). The check is a host-side read that does not synchronize, and `cudaGetLastError()` clears the slot, so a launch error is reported at its own launch and never again by a later, unrelated check. Errors of a running kernel are not seen by the launch check: they surface at the first CUDA call made after the fault, at the latest at the next synchronizing call; run with `CUDA_LAUNCH_BLOCKING=1` or under `compute-sanitizer` to locate them. Release calls (`cudaFree`, `cudaFreeHost`, `*Destroy`) run in destructors and `release()` paths and must not throw: wrap them in `ignore_release_error(...)`, which clears the error a failed release leaves in the slot and counts it
- **No allocation or default-stream copies inside the propagator loop**: `cudaMalloc`/`cudaFree` and synchronous `cudaMemcpy` serialize all streams. Tables rebuilt after a box change reuse their device buffers and are uploaded with `cudaMemcpyAsync` on the object's own stream

The CUDA implementations are in `src/platforms/cuda/*.cu`. Memory-saving mode is available via `reduce_memory=True` (stores only checkpoints, increases execution time 2-4x).

**cuFFT Input Corruption Warning**: cuFFT overwrites the input buffer of complex-to-real transforms (Z2D) even when they are out-of-place. The cuFFT User's Guide (section "Data Layout") states: "Out-of-place complex-to-real FFT will always overwrite input buffer." **Always copy the input to a work buffer before a Z2D transform if the input must be preserved.** Several solvers rely on this pattern after IFFT operations. The guide says nothing about real-to-complex transforms (D2Z); measured with CUDA 11.8 on an A10 (1D/2D/3D, batch 1 and 2), D2Z left its input unchanged while Z2D overwrote it in 2D and 3D. The shared stress routine feeds the propagator pair straight into a batched D2Z and the caller reuses the pair afterwards.

## Running Simulations

### SCFT Examples

Located in `examples/scft/`:
- `Lamella3D.py`: Lamellar phase of AB diblock
- `Gyroid.py`: Gyroid phase with box relaxation
- `ABC_Triblock_Sphere3D.py`: Spherical phase of ABC triblock
- `phases/`: **Space group symmetry examples** - BCC, FCC, Gyroid, Double Diamond, Sigma, A15, and other complex phases with space group constraints for reduced memory usage and faster field operations

Run from repository root:

```bash
cd examples/scft
python Lamella3D.py
```

### Space Group Symmetry (Beta Feature)

The `examples/scft/phases/` directory contains examples using space group symmetry to exploit crystallographic periodicity. Key features:

- **Reduced basis representation**: Fields stored only at irreducible mesh points (e.g., 48-192x reduction for cubic space groups)
- **Memory savings**: Proportional to grid reduction ratio
- **Speedup**: when the group allows it, the propagator itself is computed on the reduced physical sub-grid through the crystallographic FFT (CrysFFT) fast path — Pmmm DCT (mirror planes in x/y/z), recursive 3m (3m translations) or ObliqueZ (z-mirror, any cell with α = β = 90°, e.g. hexagonal) — instead of the full-grid FFT; otherwise only the field operations shrink. Measured speedups (e.g. 3-5x for Im-3m through the 3m path, 1.1-1.4x for hexagonal groups through ObliqueZ) are tabulated in `docs/internals/FFTImplementation.md` (CrysFFT and Performance Benchmarks sections)

**Hexagonal/trigonal support**: Hexagonal and trigonal space groups (e.g., P6₃/mmc) are fully supported. Historically they produced X-shaped artifacts and biased free energies (~10⁻³) because cell-centered grid points `(i+0.5)/N` map to cell boundaries under hexagonal rotations (row sum = 0 annihilates the 0.5 offset). This was resolved in the main branch by (1) a hybrid orbit convention for hexagonal-axes settings — node-centered in-plane coordinates `i/N` with cell-centered z — making every symmetry map an exact integer operation, and (2) a group-invariant (first-Brillouin-zone, minimal-metric-form) alias representative for the k-space diffusion and stress operators, which restores exact point-group symmetry of the discretized operators on oblique cells. Space-group on/off results now agree to machine precision for all supported crystal systems. Rhombohedral-axes R settings require `nx = ny = nz` (violations are rejected with an explicit error). The `star` branch's Fourier star basis is an alternative approach and is no longer required for correctness.

Example usage in params:

```python
"space_group": {
    "symbol": "Im-3m",   # Hermann-Mauguin symbol (BCC)
    "number": 529,       # Hall number (optional)
}
```

See `examples/scft/phases/README.md` for available space groups and reduction factors.

### L-FTS Examples

Located in `examples/lfts/`:
- `Lamella.py`: Langevin dynamics of lamellar phase
- `Gyroid.py`: Fluctuating gyroid phase
- `MixtureBlockRandom.py`: Block/random copolymer mixture

### Parameter Files

Simulations are configured via Python dictionaries with keys:
- `nx`, `lx`: Grid points and box size
- `chain_model`: "discrete" or "continuous"
- `ds`: Contour discretization (typically 1/N_Ref)
- `segment_lengths`: Relative statistical segment lengths
- `chi_n`: Flory-Huggins interaction parameters × N_Ref
- `distinct_polymers`: Polymer architectures and volume fractions
- `platform`: Platform selection (see below)
- `numerical_method`: Algorithm for propagator computation
  - `"rqm4"`: RQM4 - Pseudo-spectral with 4th-order Richardson extrapolation (default)
  - `"rk2"`: RK2 - Pseudo-spectral with 2nd-order operator splitting
  - `"cn-adi2"`: CN-ADI2 - Real-space with 2nd-order Crank-Nicolson ADI

#### Platform Auto-Selection

The `platform` parameter controls which computational backend is used:
- `"cuda"`: NVIDIA GPU with cuFFT (requires CUDA)
- `"cpu-mkl"`: CPU with Intel MKL library
- `"cpu-fftw"`: CPU with FFTW library (GPL license)

**Auto-selection logic** (when `platform` is not specified):
- **1D simulations**: Uses CPU (GPU overhead outweighs benefits for small problems)
- **2D/3D simulations**: Uses `cuda` if available, otherwise falls back to CPU

## Troubleshooting

### Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| **Segmentation fault** | Set `ulimit -s unlimited` and `export OMP_STACKSIZE=1G` |
| **GPU architecture errors** | Configure with `-DPOLYMERFTS_CUDA_ARCHITECTURES=...` listing only supported compute capabilities |
| **Memory issues** | Set `"reduce_memory": True` in parameters |
| **Run on one CPU core** | Set `os.environ["OMP_MAX_ACTIVE_LEVELS"]="0"` before imports. This disables OpenMP; on CUDA it also runs the propagator streams one after another, so GPU runs keep `"1"` |
| **GPU or CPU underused** | Set `OMP_NUM_THREADS` before imports: it is the number of propagators computed concurrently (CPU threads, or CUDA streams up to 4). The Python classes default it to 1 when unset; the examples use 2 with `OMP_MAX_ACTIVE_LEVELS="1"` |

### SCFT Not Converging

Reduce mixing parameters:

```python
"optimizer": {
    "name": "am",
    "mix_min": 0.01,    # reduce from default
    "mix_init": 0.01,   # reduce from default
    "start_error": 1e-2
}
```

### Debugging Test Failures

1. **Run with verbose output**: `ctest -V -R <test_name>`
2. **Check for NaN/Inf**: Add `np.isnan(result).any()` checks
3. **Verify partition function**: Call `solver.check_total_partition()` - should return `True` (every forward/backward propagator pair yields the same Q)
4. **Compare platforms**: Run same test on both `cuda` and CPU to isolate platform-specific issues
5. **Check material conservation**: Verify `np.mean(sum(phi_species))` ≈ 1.0

### Interpreting Error Messages

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `Material conservation failed: mean(sum(phi) - 1) = ...` (`ValidationError`, L-FTS/CL-FTS) | Total monomer concentration drifted from 1: field too strong, `ds` too coarse, or a propagator/FFT bug | Reduce the field amplitude or `ds`; compare CPU vs CUDA results |
| `Partition consistency check failed (forward/backward mismatch).` (`ValidationError`, from `check_total_partition()`) | Forward and backward propagators give different Q, i.e. the solver violates material conservation | Check the FFT implementation and boundary conditions; run the same case on the other platform |
| `cuFFT error code ...`, `Failed to create cuFFT D2Z plan` | GPU memory exhausted or invalid transform size | Check GPU memory with `nvidia-smi`, reduce the grid size or set `reduce_memory=True` |
| `FFTW plan creation failed`, `FFTW memory allocation failed` | Host memory allocation failure | Reduce the grid size or increase available RAM |
| `No available platform` / `Could not find platform '...'` | The build has no backend (or a CUDA-only build runs on a node without a visible GPU), or `platform` is not one of `cuda`, `cpu-mkl`, `cpu-fftw` (`auto` is resolved in Python before reaching C++) | Rebuild with a backend enabled; check the `platform` parameter |
| `Size of input w[...] (...) and 'n_grid' (...) must match.` | A field array does not have one value per grid point (or per reduced-basis point with a space group) | Flatten the field to `n_grid` (or `n_reduced_basis`) entries |

### Where to Look When Things Go Wrong

1. **Propagator issues**: the computation order in `src/common/Computation*.cpp`, the grid operations in `src/platforms/*/C*ComputationBackend.*`, the solvers in `src/platforms/*/`
2. **FFT problems**: `src/platforms/cpu/FftwFFT.cpp` or `src/platforms/cuda/CudaFFT.cu`
3. **SCFT convergence**: `src/python/scft.py`, check Anderson mixing parameters
4. **Memory errors**: Check `reduce_memory` option, verify GPU memory availability
5. **Platform differences**: Compare CPU vs CUDA outputs - should match to ~10⁻¹³

## Chain Models

This library supports two chain models with different mathematical formulations:

### Continuous Chain Model (Gaussian Chain)

Solves the **modified diffusion equation**:

$$\frac{\partial q(\mathbf{r}, s)}{\partial s} = \frac{b^2}{6} \nabla^2 q(\mathbf{r}, s) - w(\mathbf{r}) q(\mathbf{r}, s)$$

where $q(\mathbf{r}, s)$ is the chain propagator, $b$ is the statistical segment length, and $w(\mathbf{r})$ is the potential field.

**Pseudo-spectral solution** (operator splitting):

$$q(s+\Delta s) = e^{-w \Delta s/2} \cdot \mathcal{F}^{-1}\left[ e^{-b^2 |\mathbf{k}|^2 \Delta s/6} \cdot \mathcal{F}[e^{-w \Delta s/2} \cdot q(s)] \right]$$

The term $e^{-b^2 |\mathbf{k}|^2 \Delta s/6}$ is the **diffusion propagator** in Fourier space.

### Discrete Chain Model (Chapman-Kolmogorov Equation)

Solves the **Chapman-Kolmogorov integral equation** (NOT the modified diffusion equation):

$$q(\mathbf{r}, n+1) = e^{-w(\mathbf{r}) \Delta s} \int g(\mathbf{r} - \mathbf{r}') q(\mathbf{r}', n) \, d\mathbf{r}'$$

where $g(\mathbf{r})$ is the **bond function** representing the probability distribution of bond vectors.

**Bead-spring (Gaussian) bond function**:

$$g(\mathbf{r}) = \left( \frac{3}{2\pi a^2} \right)^{3/2} \exp\left( -\frac{3|\mathbf{r}|^2}{2a^2} \right)$$

**Fourier transform of bond function**:

$$\hat{g}(\mathbf{k}) = \exp\left( -\frac{a^2 |\mathbf{k}|^2}{6} \right)$$

With the convention $a^2 = b^2 \Delta s$ (where $\Delta s = 1/N$), this becomes:

$$\hat{g}(\mathbf{k}) = \exp\left( -\frac{b^2 |\mathbf{k}|^2 \Delta s}{6} \right)$$

**Key distinction**: The formula $\exp(-b^2 |\mathbf{k}|^2 \Delta s/6)$ appears in both models but has different physical meanings:
- **Continuous**: Diffusion propagator from solving the modified diffusion equation
- **Discrete**: Bond function (Fourier transform of Gaussian bond distribution)

**Reference**: Park et al., *J. Chem. Phys.* **2019**, 150, 234901

## Numerical Methods

This library provides multiple numerical methods for solving the modified diffusion equation, each with different accuracy-performance tradeoffs.

### Pseudo-Spectral Methods (Fourier Space)

Pseudo-spectral methods use spectral transforms to solve the diffusion operator efficiently. The transform type depends on boundary conditions:
- **Periodic BC**: Fast Fourier Transform (FFT)
- **Reflecting BC**: Discrete Cosine Transform (DCT-II)
- **Absorbing BC**: Discrete Sine Transform (DST-II)

All pseudo-spectral methods use **cell-centered grids** where grid points are at $x_i = (i + 0.5) \cdot dx$ for $i = 0, 1, ..., N-1$, with boundaries at cell faces (not grid points).

#### RQM4 (4th-order Richardson Extrapolation)

The default method combining operator splitting with Richardson extrapolation:
1. Split the operator: diffusion (Fourier space) + reaction (real space)
2. Apply Strang splitting with half-steps
3. Richardson extrapolation: $q^{(4)} = \frac{4 q_{ds/2} - q_{ds}}{3}$ for 4th-order accuracy

**Order of accuracy**: 4th-order in ds (convergence order ≈ 4.0)

**Reference**: Ranjan, Qin, Morse, *Macromolecules* **2008**, 41, 942

#### RK2 (2nd-order Rasmussen-Kalosakas)

Simple operator splitting without Richardson extrapolation (note: RK = Rasmussen-Kalosakas, not Runge-Kutta):
1. Split the operator: diffusion (Fourier space) + reaction (real space)
2. Apply Strang splitting: $q(s+\Delta s) = e^{-w \Delta s/2} \cdot \mathcal{F}^{-1}[ e^{-k^2 b^2 \Delta s/6} \cdot \mathcal{F}[e^{-w \Delta s/2} \cdot q(s)] ]$

**Order of accuracy**: 2nd-order in ds (convergence order ≈ 2.0)

**Performance**: Faster than RQM4 (2 FFTs vs 6 FFTs per step) but lower accuracy.

**Note**: RK2 for continuous chains is mathematically equivalent to the **N-bond model** for discrete chains (Park et al. 2019). Both use the same Boltzmann factor $e^{-b^2 k^2 \Delta s / 6}$ in Fourier space.

**Reference**: Rasmussen & Kalosakas, *J. Polym. Sci. B* **2002**, 40, 1777

### Real-Space Methods (Finite Difference)

Real-space methods use **cell-centered grids** (same as pseudo-spectral methods) for consistent boundary condition handling:
- **Reflecting BC**: Symmetric ghost cell ($q_{-1} = q_0$)
- **Absorbing BC**: Antisymmetric ghost cell ($q_{-1} = -q_0$)

#### CN-ADI2 (Crank-Nicolson ADI, 2nd-order)

Alternating Direction Implicit method with Crank-Nicolson time stepping:
1. Split 2D/3D problem into sequence of 1D problems
2. Each direction solved implicitly with tridiagonal system
3. 2nd-order accurate in space and time

**Order of accuracy**: 2nd-order in ds (convergence order ≈ 2.0)

### Method Selection Guidelines

| Method | Order | Material Conservation | Best For | Limitations |
|--------|-------|----------------------|----------|-------------|
| RQM4 | 4th | Exact (~10⁻¹⁶) | General use, default choice | None |
| RK2 | 2nd | Exact (~10⁻¹⁶) | Fast iterations | Lower accuracy |
| CN-ADI2 | 2nd | Exact (~10⁻¹⁵) | Fast prototyping | Lower accuracy |

All methods support periodic, reflecting, and absorbing boundary conditions.

**Note on Material Conservation**: Methods with "Exact" conservation satisfy (VU)†=VU where V is the volume matrix and U is the evolution operator. This ensures forward and backward partition functions are equal to machine precision. See Yong & Kim, *Phys. Rev. E* **2017**, 96, 063312 for the theoretical foundation.

## Units and Conventions

- Length unit: $b N^{1/2}$ where $b$ is reference statistical segment length, $N$ is reference polymerization index
- Fields: Defined as "per reference chain" potential (multiply by `ds` to get "per segment" potential)
- This follows notation in *Macromolecules* **2013**, 46, 8037

## Development Notes

### When Modifying C++ Code

1. Changes to `src/common/*.cpp` or `src/platforms/*/*.cpp|.cu` require rebuilding:

   ```bash
   cd build && make -j8 && make install
   ```

2. Platform-specific features must be implemented in both `cpu/` and `cuda/` unless truly platform-dependent. The order of a computation goes into `src/common/Computation*.cpp` once; new grid operations go into both `ComputationBackend` implementations

3. Memory management: allocations are owned by objects that release them. Device memory lives in `CudaDeviceBuffer<T>` and page-locked host memory in `CudaPinnedBuffer<T>` (`src/platforms/cuda/CudaCommon.h`), host arrays in `std::vector`, owned objects in `std::unique_ptr`, so destructors and constructors that throw need no cleanup code. Raw pointers are only lent (`.get()`, kernel and cuFFT arguments; test a buffer for null with `.empty()`); the exceptions are the factory functions, which hand ownership to the caller, the backends' `allocate()`/`deallocate()` and `allocate_checkpoint()`/`deallocate_checkpoint()` pairs, which are the allocators of the propagator handles and checkpoints, and the `CudaCommon` singleton; FFTW/MKL plans and descriptors and the `fftw_malloc` buffers of `FftwFFT`/`FftwCrysFFT*` are freed by those classes' `release()`. Every owned CUDA allocation goes through `counted_cuda_allocate()`, so the live counters and the failure hook used by `tests/TestCudaDeviceBuffer.cu` see it

4. The propagator computation optimizer (`PropagatorComputationOptimizer`) automatically detects redundant calculations using hash tables of `PropagatorCode` objects - avoid manual optimization

### Design Decisions

**CPU Pseudo-Spectral Solver Hierarchy**: `CpuSolverPseudoRQM4`, `CpuSolverPseudoRK2`, and `CpuSolverPseudoDiscrete` share common functionality through the `CpuSolverPseudoBase` base class. The base class provides:
- FFT object management (`init_shared`; the FFT and `Pseudo` objects are owning members)
- Transform dispatch (`transform_forward`/`transform_backward`)
- Laplacian operator updates (`update_laplacian_operator`)
- Stress computation with customizable coefficient factor (`compute_single_segment_stress` + `get_stress_boltz_bond`)

The CUDA side mirrors this: `CudaSolverPseudoRQM4`, `CudaSolverPseudoRK2` and `CudaSolverPseudoDiscrete` derive from `CudaSolverPseudoBase`, which owns the cuFFT plans and DCT/DST objects (`fft_forward`/`fft_backward`/`fft_`), the space-group/CrysFFT setup (index maps from the platform-independent `CrysFFTBasisMapping.h`), `update_laplacian_operator`, the shared `update_dw_impl`, and `compute_single_segment_stress` itself (the derived classes only supply `stress_scratch`, the workspaces they lend; the discrete solver also overrides `get_stress_boltz_bond`). Mechanical real/complex differences (D2Z vs Z2Z cuFFT calls, CUB sum vs reduce) go through the overloads in `CudaTypeDispatch.h`; a remaining `if constexpr` on the field type marks a genuine algorithmic difference and is commented as such.

**Computation Class Hierarchy**: the propagator computations are platform neutral and live in `src/common/`: `ComputationCommon<T>` (backend, scheduler, block/solvent concentrations, getters, stress conversion) with `ComputationContinuous<T>`, `ComputationDiscrete<T>` (full storage) and `ComputationReduceMemoryBase<T>` -> `ComputationReduceMemoryContinuous<T>`, `ComputationReduceMemoryDiscrete<T>` (checkpoints). They decide what is computed and in which order, and their loops read top to bottom in that order. Every grid operation goes through `ComputationBackend<T>` (`src/common/ComputationBackend.h`), implemented by `CpuComputationBackend<T>` (host arrays, OpenMP threads, CPU solvers) and `CudaComputationBackend<T>` (device arrays, CUDA streams, CUDA solvers, checkpoints in pinned host memory). The factories (`MklFactory`, `FftwFactory`, `CudaFactory`) create the computations with their backend; there are no platform-specific computation classes, and the C++ tests obtain computations through `tests/TestComputationFactory.h`. Backend rules: one operation is one physically meaningful step on whole fields (no array-level scale/add helpers); `BackendArray<T>` and `CheckpointArray<T>` are incomplete types, so common code cannot dereference them; reduce-memory mode reproduces full storage bit for bit (same segment pairs, weights and summation order).

Derived solver classes implement chain-model-specific behavior:
- `update_dw`: Different Boltzmann factor formulas (ds*0.5 vs ds)
- `advance_propagator`: Different algorithms (RQM4 vs simple step)
- `advance_propagator_half_bond_step`: Discrete-specific implementation
- `get_stress_boltz_bond`: the base returns nullptr (continuous chains); the discrete solver overrides it with boltz_bond

**Spectral Transform Hierarchy**: `FFT<T>` is the abstract base class for all spectral transforms. Platform implementations:

```text
      FFT<T>              (abstract base - double* and complex* interfaces)
        ↑
   FftwFFT<T, DIM>        (CPU: FFTW for FFT, DCT, DST)
   MklFFT<T, DIM>         (CPU: MKL for FFT, DCT, DST)
   CudaFFTBase<T>         (GPU abstract base, src/platforms/cuda/CudaFFT.h: adds pure virtual
        ↑                  forward_stream/backward_stream(..., cudaStream_t) overloads)
   CudaFFT<T, DIM>        (GPU: cuFFT for FFT, custom kernels for DCT/DST; forward()/backward()
                           call the stream versions on stream 0)
```

**FFT Implementation Rules** (CRITICAL - frequently violated, causes subtle bugs):

FFT functions are called from multiple OpenMP threads simultaneously (propagator-level parallelism). This requires careful implementation to avoid race conditions and data corruption.

1. **DO NOT multi-thread inside FFT**: Disable internal threading in FFTW and cuFFT. Parallelism happens at the propagator level, not inside FFT calls.

2. **DO NOT share buffers between threads**: Internal work buffers MUST be `thread_local`. The class member `work_buffer_` and `complex_buffer_` can only be used for plan creation, not execution. Example pattern:

   ```cpp
   void forward(T* rdata, double* cdata) {
       thread_local std::vector<double> work_local;  // Thread-safe
       // NOT: std::memcpy(work_buffer_, rdata, ...);  // Race condition!
   }
   ```

3. **ALWAYS preserve input arrays after transforms**: FFTW c2r and cuFFT c2r/z2d **destroy the input array** even for "out-of-place" transforms. Always copy input to a local buffer before executing when the input must be preserved across stages.

4. **Use new-array execute functions**: In FFTW, use `fftw_execute_dft_r2c`, `fftw_execute_dft_c2r`, `fftw_execute_dft` (not plain `fftw_execute`) when operating on thread-local buffers.

Violating these rules causes tests to fail with garbage values (e.g., partition function = -10^7 instead of ~1) or intermittent failures that are hard to debug.

`FFT<T>` provides both interfaces:
- `forward(T*, double*)` / `backward(double*, T*)`: Universal interface for all BCs (FFT, DCT, DST)
- `forward(T*, complex<double>*)` / `backward(complex<double>*, T*)`: Periodic BC only

`CudaFFT` additionally provides stream-aware methods for async execution:
- `forward_stream(T*, double*, cudaStream_t)` / `backward_stream(double*, T*, cudaStream_t)`
- `forward_stream(T*, complex<double>*, cudaStream_t)` / `backward_stream(complex<double>*, T*, cudaStream_t)`

CPU solvers hold `std::unique_ptr<FFT<T>> fft_` and call `fft_->forward()` directly without dimension-specific casting; the CUDA solvers hold their DCT/DST objects as `std::unique_ptr<CudaFFTBase<T>>` and call `forward_stream()`/`backward_stream()` with the propagator's stream.

### When Modifying Python Code

Changes to `src/python/*.py` are picked up by the tests after `make` (they are staged into `build/python/`) and by the installed package after `make install`. No recompilation needed.

### Deprecated/Internal Methods

**Never use `advance_propagator_single_segment`**: This is an internal low-level method that should never be used in any case. Always use `compute_propagators()` to compute all propagators at once, then access them via `get_propagator(polymer, v, u, step)`. The `PropagatorSolver` class provides a clean interface: call `compute_propagators()` to compute all propagators, then use `get_propagator()`, `get_partition_function()`, and `get_concentration()` to access results.

### Adding New Monomer Types or Interactions

Modify only the parameter dictionary - the code supports arbitrary numbers of monomer types. The `SymmetricPolymerTheory` class handles interaction matrix eigendecomposition automatically.

### Validation

Results must match:
- **PSCF** (https://github.com/dmorse/pscfpp) for continuous chains with even contour steps
- **Previous FTS studies** for discrete AB diblock (*Polymers* **2021**, 13, 2437)
- Results should be **identical across platforms** (CUDA vs CPU) within machine precision - verify by running same parameters on both platforms

## Documentation and Learning

- **Tutorials**: `tutorials/` contains Jupyter notebooks explaining theory and usage (see `tutorials/README.md` for recommended order)
- **Examples**: `examples/scft/`, `examples/lfts/` and `examples/clfts/` contain runnable simulation scripts
- **Architecture**: `docs/internals/Architecture.md` (classes, ownership and one iteration on one page)
- **API Documentation**: Can be generated with Doxygen using `Doxyfile` in root directory
- **Deep Learning Extension**: For DL-boosted L-FTS, see https://github.com/yongdd/deep-langevin-fts

### LaTeX Validation for Documentation

GitHub uses KaTeX to render LaTeX in markdown files. Before pushing documentation changes, validate LaTeX syntax:

```bash
# Requires: npm install -g katex
python scripts/check_latex.py
```

This script extracts all LaTeX blocks from `docs/*.md` and validates them with KaTeX CLI. Common issues that cause rendering failures:
- `\left\{` → use `\left\lbrace` instead
- `\right\}` → use `\right\rbrace` instead
- Unmatched `\left` and `\right` delimiters

## Key References

The implementation is based on these publications:
- Chain propagator optimization: *J. Chem. Theory Comput.* **2025**, 21, 3676
- Discrete chain model: *J. Chem. Phys.* **2019**, 150, 234901 (Park et al.)
- Multi-monomer theory: *Macromolecules* **2025**, 58, 816
- L-FTS algorithm: *Polymers* **2021**, 13, 2437
- Field update methods: *J. Chem. Phys.* **2023**, 158, 114117
- CUDA implementation: *Eur. Phys. J. E* **2020**, 43, 15
- RQM4 method: *Macromolecules* **2008**, 41, 942 (Ranjan, Qin, Morse)
- RK2 method: *J. Polym. Sci. B* **2002**, 40, 1777 (Rasmussen, Kalosakas)
- Pseudo-spectral algorithm benchmarks: *Eur. Phys. J. E* **2011**, 34, 110 (Stasiak, Matsen)
- Material conservation: *Phys. Rev. E* **2017**, 96, 063312 (Yong, Kim)
