# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a polymer field theory simulation library implementing Self-Consistent Field Theory (SCFT) and Langevin Field-Theoretic Simulations (L-FTS). The codebase consists of high-performance C++/CUDA computational kernels exposed through Python interfaces via pybind11, enabling efficient polymer physics simulations with Python's ecosystem.

## Build System

### Build Commands
```bash
# Initial build (from repository root)
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
make -j8

# Run all tests
make test

# Install Python modules to conda environment
make install

# Clean build (if needed)
cd build && rm -rf * && cmake ../ -DCMAKE_BUILD_TYPE=Release && make -j8
```

### Important Build Notes
- Requires C++20 standard (set in CMakeLists.txt)
- CUDA Toolkit 11.8+ required for GPU support
- Set `CUDA_ARCHITECTURES` in CMakeLists.txt:102 based on target GPU (default includes compute capabilities 60-90)
- If encountering "Unsupported gpu architecture" errors, remove higher compute capabilities from `CUDA_ARCHITECTURES`
- Debug builds: Change `CMAKE_BUILD_TYPE` to `Debug` for additional warnings and profiling symbols

### Environment Setup
```bash
# Activate conda environment (required before any work)
conda activate polymerfts

# Set stack size (important to avoid segfaults)
ulimit -s unlimited
export OMP_STACKSIZE=1G
```

## Testing

### C++ Tests
```bash
# From build directory
cd build

# Run all tests
make test

# Run specific test (example)
./tests/TestPseudoBranchedContinuous3D
./tests/TestScft3D

# Run tests with verbose output
ctest -V

# Run specific test pattern
ctest -R Pseudo  # runs all tests matching "Pseudo"
```

### Python Tests
```bash
# From repository root
python tests/TestNumPyFFT1D.py
python tests/TestNumPyFFT2D.py
python tests/TestNumPyFFT3D.py
```

## Code Architecture

### Platform Abstraction (Abstract Factory Pattern)

The codebase uses the **abstract factory pattern** to support multiple computational platforms (CPU with Intel MKL, CUDA GPUs). This is a key architectural feature:

- **Factory Selection**: `src/common/PlatformSelector.cpp` determines available platforms and creates appropriate factory instances
- **Platform Implementations**:
  - `src/platforms/cpu/`: MKL-based CPU implementations using Intel Math Kernel Library for FFT and linear algebra
  - `src/platforms/cuda/`: CUDA GPU implementations using cuFFT and custom CUDA kernels
- **Common Interfaces**: `src/common/AbstractFactory.h` defines abstract interfaces that all platform implementations must satisfy

When adding new computational features, you must implement them for both platforms unless explicitly platform-specific.

### Core Components

#### Propagator Computation (`src/common/PropagatorComputation.h`)
The central computational engine. Computes chain propagators using dynamic programming to avoid redundant calculations for branched polymers. The optimization strategy is described in *J. Chem. Theory Comput.* **2025**, 21, 3676.

Key concepts:
- **Chain propagators**: Solutions to modified diffusion equations (continuous) or recursive integral equations (discrete) representing polymer chain statistics
- **Continuous chains**: Pseudo-spectral method with RQM4 (4th-order Richardson extrapolation) solving the modified diffusion equation
- **Discrete chains**: Pseudo-spectral method using bond convolution based on Chapman-Kolmogorov equations (N-1 bond model from Park et al. 2019)
- **Real-space method**: CN-ADI (Crank-Nicolson ADI) finite difference solver (beta feature, continuous chains only). CN-ADI2 (2nd-order, default) or CN-ADI4 (4th-order via `-DPOLYMERFTS_USE_CN_ADI4=ON`)
- **Aggregation**: Automatic detection and reuse of equivalent propagator computations in branched/mixed polymer systems

#### Computation Box (`src/common/ComputationBox.h`)
Manages simulation grid, FFT operations, and boundary conditions. Handles 1D/2D/3D simulations with periodic boundaries (pseudo-spectral) or periodic/reflecting/absorbing boundaries (real-space).

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
  - Supports Anderson Mixing and ADAM optimizers
  - Implements stress calculations for box size optimization
  - Space group symmetry constraints (beta)

- `src/python/lfts.py`: L-FTS simulation orchestrator
  - Implements Langevin dynamics with Leimkuhler-Matthews method
  - Structure function calculations
  - Field compressors (Anderson Mixing, Linear Response)
  - PCG64 random number generator

- `src/python/polymer_field_theory.py`: Multi-monomer field theory transformations
  - Converts between monomer potential fields and auxiliary fields
  - Eigenvalue decomposition of interaction matrices
  - Hamiltonian coefficient calculations for arbitrary χN parameters

#### Python-C++ Binding (`src/pybind11/polymerfts_core.cpp`)
Exposes C++ classes to Python as the `_core` module. All C++ computational objects are accessible through this interface.

### CUDA Implementation Details

CUDA code uses:
- **Multiple streams** (up to 4): Parallel propagator computations on GPU
- **cuFFT**: GPU-accelerated Fast Fourier Transforms
- **Shared memory**: Tridiagonal solvers for real-space methods use shared memory for performance
- **Pinned memory**: Host-device transfers use pinned circular buffers for efficiency

The CUDA implementations are in `src/platforms/cuda/*.cu`. Memory-saving mode is available via `reduce_memory_usage=True` (stores only checkpoints, increases execution time 2-4x).

## Running Simulations

### SCFT Examples
Located in `examples/scft/`:
- `Lamella3D.py`: Lamellar phase of AB diblock
- `Gyroid.py`: Gyroid phase with box relaxation
- `ABC_Triblock_Sphere3D.py`: Spherical phase of ABC triblock
- `phases/`: Various morphologies (cylinder, perforated lamella, double diamond, etc.)

Run from repository root:
```bash
cd examples/scft
python Lamella3D.py
```

### L-FTS Examples
Located in `examples/fts/`:
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
- `platform`: "cuda" or "cpu-mkl" (auto-selected by default: cuda for 2D/3D, cpu-mkl for 1D)

### Common Issues and Solutions

**Segmentation fault**: Set `ulimit -s unlimited` and `export OMP_STACKSIZE=1G`

**SCFT not converging**: Reduce mixing parameters:
```python
"optimizer": {
    "name": "am",
    "mix_min": 0.01,    # reduce from default
    "mix_init": 0.01,   # reduce from default
    "start_error": 1e-2
}
```

**GPU architecture errors**: Edit `CMakeLists.txt:102` to remove unsupported compute capabilities from `CUDA_ARCHITECTURES`

**Memory issues**: Set `"reduce_memory_usage": True` in parameters

**Single CPU core usage**: Set `os.environ["OMP_MAX_ACTIVE_LEVELS"]="0"` before imports

## Units and Conventions

- Length unit: $b N^{1/2}$ where $b$ is reference statistical segment length, $N$ is reference polymerization index
- Fields: Defined as "per reference chain" potential (multiply by `ds` to get "per segment" potential)
- This follows notation in *Macromolecules* **2013**, 46, 8037

## Development Notes

### Workflow Rules

- **Never commit without permission**: Always wait for explicit user approval before running `git commit`.

### When Modifying C++ Code

1. Changes to `src/common/*.cpp` or `src/platforms/*/*.cpp|.cu` require rebuilding:
   ```bash
   cd build && make -j8 && make install
   ```

2. Platform-specific features must be implemented in both `cpu/` and `cuda/` unless truly platform-dependent

3. Memory management: C++ uses raw pointers; ensure proper allocation/deallocation in constructors/destructors

4. The propagator computation optimizer (`PropagatorComputationOptimizer`) automatically detects redundant calculations using hash tables of `PropagatorCode` objects - avoid manual optimization

### Design Decisions

**CPU Pseudo-Spectral Solver Hierarchy**: `CpuSolverPseudoContinuous` and `CpuSolverPseudoDiscrete` share common functionality through the `CpuSolverPseudoBase` base class. The base class provides:
- FFT object management (`init_shared`/`cleanup_shared`)
- Transform dispatch (`transform_forward`/`transform_backward`)
- Laplacian operator updates (`update_laplacian_operator`)
- Stress computation with customizable coefficient factor (`compute_single_segment_stress` + `get_stress_boltz_bond`)

Derived classes implement chain-model-specific behavior:
- `update_dw`: Different Boltzmann factor formulas (ds*0.5 vs ds)
- `advance_propagator`: Different algorithms (RQM4 vs simple step)
- `advance_propagator_half_bond_step`: Discrete-specific implementation
- `get_stress_boltz_bond`: Returns nullptr for continuous, boltz_bond for discrete

**Spectral Transform Hierarchy**: `FFT<T>` is the abstract base class for all spectral transforms. Platform implementations:
```
      FFT<T>              (abstract base - double* and complex* interfaces)
        ↑
   MklFFT<T, DIM>         (CPU: Intel MKL for FFT, DCT, DST)
   CudaFFT<T, DIM>        (GPU: cuFFT for FFT, custom kernels for DCT/DST)
```
`FFT<T>` provides both interfaces:
- `forward(T*, double*)` / `backward(double*, T*)`: Universal interface for all BCs (FFT, DCT, DST)
- `forward(T*, complex<double>*)` / `backward(complex<double>*, T*)`: Periodic BC only

`CudaFFT` additionally provides stream-aware methods for async execution:
- `forward_stream(T*, double*, cudaStream_t)` / `backward_stream(double*, T*, cudaStream_t)`
- `forward_stream(T*, complex<double>*, cudaStream_t)` / `backward_stream(complex<double>*, T*, cudaStream_t)`

Solvers store `FFT<T>* fft_` and call `fft_->forward()` directly without dimension-specific casting.

### When Modifying Python Code

Changes to `src/python/*.py` take effect after `make install` from build directory. No recompilation needed.

### Deprecated/Internal Methods

**Never use `advance_propagator_single_segment`**: This is an internal low-level method that should never be used in any case. Always use `compute_propagators()` to compute all propagators at once, then access them via `get_chain_propagator(polymer, v, u, step)`. The `PropagatorSolver` class provides a clean interface: call `compute_propagators()` to compute all propagators, then use `get_propagator()`, `get_partition_function()`, and `get_concentration()` to access results.

### Adding New Monomer Types or Interactions

Modify only the parameter dictionary - the code supports arbitrary numbers of monomer types. The `SymmetricPolymerTheory` class handles interaction matrix eigendecomposition automatically.

### Validation

Results must match:
- **PSCF** (https://github.com/dmorse/pscfpp) for continuous chains with even contour steps
- **Previous FTS studies** for discrete AB diblock (*Polymers* **2021**, 13, 2437)
- Results should be **identical across platforms** (CUDA vs MKL) within machine precision - verify by running same parameters on both platforms

## Documentation and Learning

- **Tutorials**: `tutorials/` contains Jupyter notebooks explaining theory and usage (see `tutorials/README.md` for recommended order)
- **Examples**: `examples/scft/` and `examples/fts/` contain runnable simulation scripts
- **API Documentation**: Can be generated with Doxygen using `Doxyfile` in root directory
- **Deep Learning Extension**: For DL-boosted L-FTS, see https://github.com/yongdd/deep-langevin-fts

## Key References

The implementation is based on these publications:
- Chain propagator optimization: *J. Chem. Theory Comput.* **2025**, 21, 3676
- Discrete chain model: *J. Chem. Phys.* **2019**, 150, 234901 (Park et al.)
- Multi-monomer theory: *Macromolecules* **2025**, 58, 816
- L-FTS algorithm: *Polymers* **2021**, 13, 2437
- Field update methods: *J. Chem. Phys.* **2023**, 158, 114117
- CUDA implementation: *Eur. Phys. J. E* **2020**, 43, 15
- RQM4 method: *Macromolecules* **2008**, 41, 942 (Ranjan, Qin, Morse)
- Pseudo-spectral algorithm benchmarks: *Eur. Phys. J. E* **2011**, 34, 110 (Stasiak, Matsen)
- ETDRK4 method: *Chinese J. Polym. Sci.* **2018**, 36, 488 (Song, Liu, Zhang)
