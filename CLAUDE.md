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

### Performance and Accuracy Benchmarks

Benchmark scripts for numerical method comparison are in `tests/`:
- `tests/benchmark_numerical_methods.py`: Comprehensive benchmark comparing RQM4, ETDRK4, CN-ADI2, CN-ADI4
- `tests/benchmark_etdrk4_vs_rqm4.py`: Focused comparison of pseudo-spectral methods

**Running benchmarks with SLURM job scheduler:**

Use the provided SLURM template `slurm_run.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=a10          # GPU partition (adjust for your cluster)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=4        # CPUs for OpenMP threads
#SBATCH --time=02:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.out

export OMP_MAX_ACTIVE_LEVELS=0
export OMP_NUM_THREADS=4

python -u tests/benchmark_numerical_methods.py
```

Submit the job:
```bash
sbatch slurm_run.sh
```

**Benchmark output:**
- Convergence analysis (Q vs ds) following Stasiak & Matsen methodology
- Performance comparison (time vs N) following Song et al. methodology
- Cross-platform consistency verification (CPU vs CUDA)
- Results saved to `benchmark_results.json`

**Key metrics to verify:**
- RQM4 convergence order ≈ 4.0
- CN-ADI2 convergence order ≈ 2.0
- CPU vs CUDA results identical within machine precision (~10⁻¹³)

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
- **Continuous chains**: Pseudo-spectral method with RQM4 or ETDRK4 solving the modified diffusion equation
- **Discrete chains**: Pseudo-spectral method using bond convolution based on Chapman-Kolmogorov equations (N-1 bond model from Park et al. 2019)
- **Real-space method**: CN-ADI (Crank-Nicolson ADI) finite difference solver (beta feature, continuous chains only). CN-ADI2 (2nd-order), CN-ADI4 (4th-order), or SDC (Spectral Deferred Correction)
- **Numerical method selection**: Runtime selection via `numerical_method` parameter: `"rqm4"` (RQM4), `"etdrk4"` (ETDRK4), `"cn-adi2"` (CN-ADI2), `"cn-adi4-lr"` (CN-ADI4-LR), or `"sdc"` (SDC)
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

**cuFFT Input Corruption Warning**: cuFFT may corrupt the input buffer even for out-of-place transforms, particularly for Z2D (complex-to-real) and D2Z (real-to-complex) operations. This is documented NVIDIA behavior. **Always copy input data to a work buffer before calling cuFFT if the input must be preserved.** This issue was discovered in the ETDRK4 solver where Fourier coefficients were corrupted after IFFT operations. See `CudaSolverPseudoETDRK4.cu` for the correct pattern using `cudaMemcpyAsync` to preserve input data.

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
- `numerical_method`: Algorithm for propagator computation
  - `"rqm4"`: RQM4 - Pseudo-spectral with 4th-order Richardson extrapolation
  - `"etdrk4"`: ETDRK4 - Pseudo-spectral with Exponential Time Differencing RK4
  - `"cn-adi2"`: CN-ADI2 - Real-space with 2nd-order Crank-Nicolson ADI
  - `"cn-adi4-lr"`: CN-ADI4-LR - Real-space with 4th-order CN-ADI (Local Richardson extrapolation)
  - `"sdc"`: SDC - Real-space with Spectral Deferred Correction (Gauss-Lobatto)

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

#### RQM4 (4th-order Richardson Extrapolation)
The default method combining operator splitting with Richardson extrapolation:
1. Split the operator: diffusion (Fourier space) + reaction (real space)
2. Apply Strang splitting with half-steps
3. Richardson extrapolation: $q^{(4)} = \frac{4 q_{ds/2} - q_{ds}}{3}$ for 4th-order accuracy

**Order of accuracy**: 4th-order in ds (convergence order ≈ 4.0)

**Reference**: Ranjan, Qin, Morse, *Macromolecules* **2008**, 41, 942

#### ETDRK4 (Exponential Time Differencing Runge-Kutta 4)
Direct integration using matrix exponentials:
1. Transform to Fourier space
2. Solve using exponential integrating factor
3. 4th-order Runge-Kutta for the nonlinear term

**Order of accuracy**: 4th-order in ds

**Reference**: Song, Liu, Zhang, *Chinese J. Polym. Sci.* **2018**, 36, 488

### Real-Space Methods (Finite Difference)

#### CN-ADI2 (Crank-Nicolson ADI, 2nd-order)
Alternating Direction Implicit method with Crank-Nicolson time stepping:
1. Split 2D/3D problem into sequence of 1D problems
2. Each direction solved implicitly with tridiagonal system
3. 2nd-order accurate in space and time

**Order of accuracy**: 2nd-order in ds (convergence order ≈ 2.0)

#### CN-ADI4-LR (4th-order Local Richardson Extrapolation)
Richardson extrapolation applied to CN-ADI2 at each contour step:
$$q^{(4)} = \frac{4 q_{ds/2} - q_{ds}}{3}$$

**Order of accuracy**: 4th-order in ds

"LR" stands for "Local Richardson" - the extrapolation is applied independently at each step.

#### SDC (Spectral Deferred Correction)

SDC is an iterative method that uses spectral quadrature to achieve high-order accuracy. The implementation uses **Gauss-Lobatto collocation nodes** and **IMEX (Implicit-Explicit) splitting**.

**Algorithm per contour step:**

1. **Discretize** the interval $[s_n, s_{n+1}]$ using $M$ Gauss-Lobatto nodes $\tau_m \in [0, 1]$:
   - For $M=3$: $\tau = [0, 0.5, 1]$
   - For $M=4$: $\tau = [0, 0.276..., 0.724..., 1]$

2. **Predictor** (Backward Euler at each sub-interval):
   - Solve implicitly: $(I - \Delta\tau \cdot D\nabla^2) X^{[0]}_{m+1} = e^{-w\Delta\tau} X^{[0]}_m$
   - Uses ADI for multi-dimensional cases

3. **Corrector** ($K$ iterations):
   - Compute $F_m = D\nabla^2 q_m - w \cdot q_m$ at all nodes
   - Apply spectral integral: $X^{[k+1]}_{m+1} = X^{[k]}_m + \int_0^{\tau_{m+1}} L(F^{[k]}) d\tau' - \Delta\tau \cdot F^{[k]}_{m+1}$
   - where $L$ is Lagrange interpolation through the $F$ values

**Spectral Integration Matrix:**

The integration is performed using the matrix $S$ where $S_{m,j}$ gives the contribution of $F_j$ to the integral from $\tau_0$ to $\tau_m$:
$$\int_0^{\tau_m} \sum_j L_j(\tau') F_j \, d\tau' = \sum_j S_{m,j} F_j$$

**Order of Accuracy:**

- **1D**: The method achieves high order (up to order $2K+1$ with $K$ corrections) because the implicit solves are exact
- **2D/3D**: **Limited to 2nd-order** due to $O(\Delta s^2)$ splitting error from ADI

The ADI splitting introduces an irreducible $O(\Delta s^2)$ error in 2D/3D that does not decrease with more SDC corrections. This is because ADI solves:
$$(I - \Delta\tau D_x)(I - \Delta\tau D_y) q = \text{RHS}$$
instead of the exact:
$$(I - \Delta\tau (D_x + D_y)) q = \text{RHS}$$

The difference is $O(\Delta\tau^2 D_x D_y)$, which persists regardless of the number of corrections.

**Configuration:**
- `M`: Number of Gauss-Lobatto nodes (default: 3)
- `K`: Number of correction iterations (default: 2)

**References:**
- Dutt, Greengard, Rokhlin, *BIT Numerical Mathematics* **2000**, 40, 241
- Minion, *Commun. Math. Sci.* **2003**, 1, 471

### Method Selection Guidelines

| Method | Order | Best For | Limitations |
|--------|-------|----------|-------------|
| RQM4 | 4th | General use, moderate accuracy | Requires periodic BCs |
| ETDRK4 | 4th | High accuracy pseudo-spectral | Requires periodic BCs |
| CN-ADI2 | 2nd | Non-periodic BCs, fast | Lower accuracy |
| CN-ADI4-LR | 4th | Non-periodic BCs, accuracy | 2× cost of CN-ADI2 |
| SDC | 2nd* | Experimental | *Limited by ADI in 2D/3D |

## Units and Conventions

- Length unit: $b N^{1/2}$ where $b$ is reference statistical segment length, $N$ is reference polymerization index
- Fields: Defined as "per reference chain" potential (multiply by `ds` to get "per segment" potential)
- This follows notation in *Macromolecules* **2013**, 46, 8037

## Development Notes

### Workflow Rules

- **Never commit without permission**: Always wait for explicit user approval before running `git commit`. The user must explicitly say "commit" or "make a commit" - do NOT interpret "update", "add", "change", or "fix" as permission to commit. After making changes, ask "Should I commit this change?" and wait for confirmation.

- **Use SLURM for long-running jobs**: For any computation expected to take longer than 2 minutes, submit it as a SLURM job instead of running directly. This includes benchmarks, SCFT convergence tests, and parameter sweeps. Use `sbatch` to submit jobs and launch multiple jobs simultaneously when running parameter studies or benchmarks with different configurations. Example:
  ```bash
  # Submit multiple jobs in parallel
  for param in 100 200 400 800; do
      sbatch --job-name="test_$param" --wrap="python script.py --param $param"
  done
  ```
  See `tests/submit_fig1_benchmarks.sh` for a complete example of parallel job submission.

### When Modifying C++ Code

1. Changes to `src/common/*.cpp` or `src/platforms/*/*.cpp|.cu` require rebuilding:
   ```bash
   cd build && make -j8 && make install
   ```

2. Platform-specific features must be implemented in both `cpu/` and `cuda/` unless truly platform-dependent

3. Memory management: C++ uses raw pointers; ensure proper allocation/deallocation in constructors/destructors

4. The propagator computation optimizer (`PropagatorComputationOptimizer`) automatically detects redundant calculations using hash tables of `PropagatorCode` objects - avoid manual optimization

### Design Decisions

**CPU Pseudo-Spectral Solver Hierarchy**: `CpuSolverPseudoRQM4` and `CpuSolverPseudoDiscrete` share common functionality through the `CpuSolverPseudoBase` base class. The base class provides:
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
- SDC method: *BIT Numerical Mathematics* **2000**, 40, 241 (Dutt, Greengard, Rokhlin); *Commun. Math. Sci.* **2003**, 1, 471 (Minion)
