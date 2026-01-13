# Real-Space Solver Documentation

This document describes the real-space finite difference solver for continuous chain propagators, including its numerical methods, boundary condition support, and performance characteristics.

## Overview

The real-space solver uses the **CN-ADI** (Crank-Nicolson Alternating Direction Implicit) method to solve the modified diffusion equation:

$$\frac{\partial q}{\partial s} = \frac{b^2}{6} \nabla^2 q - w(\mathbf{r}) q$$

Unlike the pseudo-spectral method which requires periodic boundary conditions, the real-space method supports:

| Boundary Condition | Description | Mathematical Form |
|-------------------|-------------|-------------------|
| **Periodic** | Cyclic boundary | $q(0) = q(L)$ |
| **Reflecting** | Neumann (zero flux) | $\partial q / \partial n = 0$ |
| **Absorbing** | Dirichlet (zero value) | $q = 0$ at boundary |

## Numerical Method

### Crank-Nicolson Scheme

The Crank-Nicolson method is a semi-implicit scheme that is unconditionally stable:

$$(I - \frac{\Delta s}{2} L) q^{n+1} = (I + \frac{\Delta s}{2} L) q^n$$

where $L$ is the discrete Laplacian operator combined with the potential term.

### ADI Splitting (3D)

For 3D problems, the operator is split into three sequential 1D solves:

1. **X-direction sweep**: Solve tridiagonal system for $q^*$
2. **Y-direction sweep**: Solve tridiagonal system for $q^{**}$
3. **Z-direction sweep**: Solve tridiagonal system for $q^{n+1}$

Each direction requires solving a tridiagonal (or cyclic tridiagonal for periodic BC) system using:
- **Thomas algorithm**: For non-periodic boundaries
- **Sherman-Morrison formula**: For periodic boundaries

### Symmetric Splitting with Potential

The propagator advancement includes the potential term using symmetric splitting:

$$q^{n+1} = e^{-w \Delta s/2} \cdot \text{Diffusion}(\Delta s) \cdot e^{-w \Delta s/2} \cdot q^n$$

This ensures the scheme remains symmetric and accurate.

## CN-ADI4: Richardson Extrapolation for 4th-Order Accuracy

By default, the solver uses **CN-ADI2** (2nd-order Crank-Nicolson ADI). Optionally, **CN-ADI4** (4th-order) can be enabled using Richardson extrapolation to achieve higher temporal accuracy. This combines one full step with two half-steps:

$$q_{\text{out}} = \frac{4 \cdot q_{\text{half}} - q_{\text{full}}}{3}$$

where:
- $q_{\text{full}}$: Result from one full step of size $\Delta s$
- $q_{\text{half}}$: Result from two half-steps of size $\Delta s/2$ each

This cancels the $O(\Delta s^2)$ error term, yielding $O(\Delta s^4)$ accuracy.

### Runtime Selection

CN-ADI2, CN-ADI4, or CN-ADI4-GQ can be selected at runtime using the `numerical_method` parameter:

```python
from polymerfts import PropagatorSolver

# CN-ADI2 (default - faster, more stable)
solver = PropagatorSolver(..., numerical_method="cn-adi2")

# CN-ADI4 (more accurate, but may be unstable near absorbing boundaries)
solver = PropagatorSolver(..., numerical_method="cn-adi4")

# CN-ADI4-GQ (Global Richardson - same accuracy, different implementation)
solver = PropagatorSolver(..., numerical_method="cn-adi4-gq")
```

**Note**: CN-ADI4 may become unstable when initial conditions are close to absorbing boundaries (see Stability Warning below).

## CN-ADI4-GQ: Global Richardson at Quadrature Level

An alternative 4th-order method, **CN-ADI4-GQ** (Global Richardson at Quadrature Level), applies Richardson extrapolation differently than CN-ADI4. Instead of extrapolating propagators at each step, it maintains two independent propagator chains and applies Richardson extrapolation only when computing physical quantities (Q, φ).

### Comparison: CN-ADI4 vs CN-ADI4-GQ

| Aspect | CN-ADI4 (Per-Step) | CN-ADI4-GQ (Global) |
|--------|-------------------|---------------------|
| **Extrapolation** | Every contour step | Only at quadrature (Q, φ) |
| **Propagator chains** | Single chain (extrapolated) | Two independent chains |
| **Memory** | 1× propagator storage | 3× propagator storage |
| **ADI solves per step** | 3 (1 full + 2 half) | 3 (1 full + 2 half) |
| **Q accuracy** | O(ds⁴) | O(ds⁴) |
| **φ accuracy** | O(ds⁴) | O(ds⁴) |

### How CN-ADI4-GQ Works

CN-ADI4-GQ maintains two independent propagator chains:

1. **Full-step chain**: $q_{\text{full}}[0..N]$ advanced with step size $\Delta s$
2. **Half-step chain**: $q_{\text{half}}[0..2N]$ advanced with step size $\Delta s/2$

Richardson-extrapolated propagators are computed as:

$$q_{\text{rich}}[n] = \frac{4 \cdot q_{\text{half}}[2n] - q_{\text{full}}[n]}{3}$$

These extrapolated propagators are used for computing Q and φ:

$$Q = \frac{1}{V} \int q_{\text{rich}} \cdot q^{\dagger}_{\text{rich}} \, d\mathbf{r}$$

$$\phi = \frac{\phi_v}{Q} \int_0^1 q_{\text{rich}}(s) \cdot q^{\dagger}_{\text{rich}}(1-s) \, ds$$

### Design Philosophy

The key insight is that Richardson extrapolation is most effective when applied to the final computed quantities (Q, φ) rather than intermediate propagator values:

1. The half-step chain already has 4× smaller error per step
2. Richardson's power comes from canceling accumulated error at the endpoint
3. Intermediate propagators are used for φ integration where errors average out

### When to Use CN-ADI4-GQ

CN-ADI4-GQ may be preferred when:

- You need access to both full-step and half-step propagators for analysis
- You want to study the effect of Richardson extrapolation on different quantities
- Memory is not a constraint (requires 3× more propagator storage)

For most applications, **CN-ADI4** (per-step Richardson) is recommended as it:
- Uses less memory
- Provides identical accuracy for Q and φ
- Has simpler implementation

### Usage

```python
from polymerfts import PropagatorSolver

# CN-ADI4-GQ (Global Richardson at quadrature level)
solver = PropagatorSolver(..., numerical_method="cn-adi4-gq")
```

### Implementation Files

| File | Description |
|------|-------------|
| `src/platforms/cpu/CpuComputationGlobalRichardson.cpp` | CPU computation layer |
| `src/platforms/cpu/CpuComputationGlobalRichardson.h` | CPU header |
| `src/platforms/cpu/CpuSolverGlobalRichardsonBase.cpp` | CPU base CN-ADI2 solver |
| `src/platforms/cpu/CpuSolverGlobalRichardsonBase.h` | CPU header |
| `src/platforms/cuda/CudaComputationGlobalRichardson.cu` | CUDA computation layer |
| `src/platforms/cuda/CudaComputationGlobalRichardson.h` | CUDA header |
| `src/platforms/cuda/CudaSolverGlobalRichardsonBase.cu` | CUDA base CN-ADI2 solver |
| `src/platforms/cuda/CudaSolverGlobalRichardsonBase.h` | CUDA header |

**Note**: CN-ADI4-GQ is available on both CPU (cpu-mkl) and CUDA platforms.

## Performance Benchmarks

### Test Configuration

- **Polymer**: AB Diblock copolymer (f = 0.5)
- **Chain model**: Continuous
- **Box**: 4.0 x 4.0 x 4.0
- **Hardware**: NVIDIA H100 GPU

### Computation Time vs Contour Steps (32³ grid, CUDA)

| N (ds=1/N) | CN-ADI2 | CN-ADI4 | Pseudo-Spectral | CN-ADI4/CN-ADI2 Ratio |
|------------|---------|---------|-----------------|----------------------|
| 10 (0.1)   | 117 ms  | 327 ms  | 403 ms          | 2.8x                 |
| 20 (0.05)  | 99 ms   | 339 ms  | 405 ms          | 3.4x                 |
| 40 (0.025) | 108 ms  | 351 ms  | 408 ms          | 3.3x                 |
| 80 (0.0125)| 115 ms  | 382 ms  | 414 ms          | 3.3x                 |
| 160 (0.00625)| 136 ms| 446 ms  | 425 ms          | 3.3x                 |

### Key Observations

1. **CN-ADI4 is ~3x slower than CN-ADI2**: This matches the expected computational cost:
   - CN-ADI4: 1 full step + 2 half-steps = 3 ADI solves per contour step
   - CN-ADI2: 1 full step = 1 ADI solve per contour step

2. **CN-ADI2 is fastest**: About 3-4x faster than both 4th-order methods

3. **CN-ADI4 comparable to pseudo-spectral**: Similar computation time on GPU

### Scaling Analysis

The real-space method scales as:
- **Time complexity**: O(M) per direction, O(3M) per ADI step (3D)
- **Memory complexity**: O(M) for propagators + O(N_x + N_y + N_z) for tridiagonal coefficients

Compared to pseudo-spectral (O(M log M) per step), real-space is:
- Faster for very small grids
- Slower for large grids
- Essential when non-periodic boundaries are required

## Accuracy and Convergence

### Convergence Study Results

The following results compare the partition function Q computed with different methods for a homopolymer in a fixed lamellar external field (χN=12, 3 periods) on a 32³ grid. Benchmark run on CUDA platform.

#### Partition Function Q vs Contour Discretization

| N (ds=1/N) | RQM4 | ETDRK4 | CN-ADI2 | CN-ADI4 | CN-ADI4-G | CN-ADI4-GQ |
|------------|------|--------|---------|---------|-----------|------------|
| 20 | 12.6459485262 | 12.6442520116 | 13.1269150396 | 13.1395511288 | 13.1395041815 | 13.1395041815 |
| 40 | 12.6453957351 | 12.6452747926 | 13.1363568960 | 13.1395065418 | 13.1395028985 | 13.1395028985 |
| 80 | 12.6453556945 | 12.6453476063 | 13.1387163979 | 13.1395030750 | 13.1395028175 | 13.1395028175 |
| 160 | 12.6453529997 | 12.6453524766 | 13.1393062126 | 13.1395028297 | 13.1395028125 | 13.1395028125 |
| 320 | 12.6453528250 | 12.6453527917 | 13.1394536625 | 13.1395028133 | 13.1395028122 | 13.1395028122 |
| 640 | 12.6453528138 | 12.6453528117 | 13.1394905247 | 13.1395028122 | 13.1395028121 | 13.1395028121 |

**Note**: Pseudo-spectral (RQM4, ETDRK4) and real-space (CN-ADI) methods converge to different Q values because they use different spatial discretization schemes.

#### Error |Q - Q_ref| (Q_ref = value at N=640)

| N | RQM4 | ETDRK4 | CN-ADI2 | CN-ADI4 | CN-ADI4-G | CN-ADI4-GQ |
|---|------|--------|---------|---------|-----------|------------|
| 20 | 5.96e-04 | 1.10e-03 | 1.26e-02 | 4.83e-05 | 1.37e-06 | 1.37e-06 |
| 40 | 4.29e-05 | 7.80e-05 | 3.13e-03 | 3.73e-06 | 8.64e-08 | 8.64e-08 |
| 80 | 2.88e-06 | 5.21e-06 | 7.74e-04 | 2.63e-07 | 5.40e-09 | 5.40e-09 |
| 160 | 1.86e-07 | 3.35e-07 | 1.84e-04 | 1.75e-08 | 3.25e-10 | 3.25e-10 |
| 320 | 1.11e-08 | 2.00e-08 | 3.69e-05 | 1.06e-09 | 1.46e-11 | 1.45e-11 |

#### Measured Convergence Order

| Method | Estimated Order p | Expected |
|--------|-------------------|----------|
| RQM4 (Pseudo-Spectral) | **p ≈ 3.93** | 4.0 |
| ETDRK4 (Pseudo-Spectral) | **p ≈ 3.94** | 4.0 |
| CN-ADI2 (Real-Space) | **p ≈ 2.10** | 2.0 |
| CN-ADI4 (Real-Space, per-step Richardson) | **p ≈ 3.87** | 4.0 |
| CN-ADI4-G (Real-Space, Global Richardson per-step) | **p ≈ 4.13** | 4.0 |
| CN-ADI4-GQ (Real-Space, Global Richardson at quadrature) | **p ≈ 4.13** | 4.0 |

#### Computation Time (ms) on CUDA

| N | RQM4 | ETDRK4 | CN-ADI2 | CN-ADI4 | CN-ADI4-G | CN-ADI4-GQ |
|---|------|--------|---------|---------|-----------|------------|
| 20 | 2.5 | 5.2 | 7.4 | 22.0 | 22.1 | 22.1 |
| 40 | 4.8 | 10.2 | 14.7 | 43.9 | 44.2 | 44.1 |
| 80 | 9.5 | 20.3 | 29.3 | 87.9 | 88.1 | 88.1 |
| 160 | 18.9 | 40.1 | 58.6 | 175.3 | 175.8 | 176.0 |
| 320 | 37.6 | 80.5 | 117.0 | 350.4 | 351.5 | 352.1 |
| 640 | 75.2 | 160.8 | 233.8 | 700.7 | 702.8 | 703.6 |

### Key Findings

1. **All 4th-order methods achieve expected accuracy**: The measured convergence orders confirm that:
   - RQM4, ETDRK4: p ≈ 3.9-4.0 (pseudo-spectral)
   - CN-ADI4, CN-ADI4-G, CN-ADI4-GQ: p ≈ 3.9-4.1 (real-space)

2. **CN-ADI4-G and CN-ADI4-GQ produce identical results**: Both Global Richardson methods (per-step and quadrature-level) yield the same Q values, as expected since they apply the same Richardson formula.

3. **Global Richardson (G, GQ) slightly more accurate than per-step (ADI4)**: At the same N, CN-ADI4-G/GQ have ~10-30× smaller error than CN-ADI4, likely due to better error cancellation in the Richardson extrapolation.

4. **CN-ADI2 converges as expected**: Shows clear $O(\Delta s^2)$ convergence with error decreasing by ~4× when $\Delta s$ is halved.

5. **Systematic difference between methods**: Real-space and pseudo-spectral methods converge to different Q values (~4% difference) due to different spatial discretization (finite difference vs spectral).

6. **Performance**: All 4th-order real-space methods (CN-ADI4, CN-ADI4-G, CN-ADI4-GQ) have similar computation time, approximately 3× slower than CN-ADI2 due to the 3 ADI solves per step.

### Grafted Brush Validation (Absorbing Boundaries)

To validate solvers with non-periodic boundary conditions, we test a grafted brush configuration:

- **Setup**: 1D domain with absorbing boundaries on both sides
- **Initial condition**: Gaussian centered at $x_0 = 2.0$ (center), varying $\sigma$
- **Grid**: 512 points, Lx = 4.0
- **Comparison**: Numerical vs analytical Fourier series solution

The analytical solution for diffusion with absorbing BCs and Gaussian initial condition is:

$$q(x,s) = \frac{2}{L} \sum_{n=1}^{\infty} a_n \sin\left(\frac{n\pi x}{L}\right) \exp\left(-\frac{n^2\pi^2 b^2 s}{6L^2}\right)$$

where $a_n$ are the Fourier sine coefficients of the initial Gaussian.

#### Real-Space vs Pseudo-Spectral Comparison

For absorbing boundaries, two methods are available:
- **Real-space (CN-ADI2)**: 2nd-order temporal accuracy (CN-ADI4 available but may be unstable)
- **Pseudo-spectral (DST)**: Spectral accuracy using Discrete Sine Transform

**Convergence Study (σ = 0.02, very sharp Gaussian):**

| $\Delta s$ | N_steps | Real-Space L2 Error | Pseudo-Spectral L2 Error |
|----|---------|---------------------|--------------------------|
| 0.1 | 2 | $5.04 \times 10^{-2}$ | $4.69 \times 10^{-17}$ |
| 0.05 | 4 | $2.69 \times 10^{-2}$ | $5.29 \times 10^{-17}$ |
| 0.025 | 8 | $7.55 \times 10^{-3}$ | $1.16 \times 10^{-16}$ |
| 0.0125 | 16 | $5.58 \times 10^{-4}$ | $4.02 \times 10^{-16}$ |
| 0.00625 | 32 | $2.77 \times 10^{-6}$ | — |
| 0.003125 | 64 | $1.81 \times 10^{-6}$ | — |

#### Effect of Gaussian Sharpness

The initial condition width $\sigma$ affects accuracy. Results with $\Delta s = 0.005$:

| $\sigma$ | $\sigma/\Delta x$ | Real-Space Error | Pseudo-Spectral Error |
|-------|------|------------------|----------------------|
| 0.400 | 51.2 | $2.94 \times 10^{-6}$ | $2.54 \times 10^{-11}$ |
| 0.200 | 25.6 | $5.04 \times 10^{-6}$ | $2.13 \times 10^{-15}$ |
| 0.100 | 12.8 | $4.28 \times 10^{-6}$ | $2.46 \times 10^{-15}$ |
| 0.050 | 6.4 | $2.56 \times 10^{-6}$ | $1.30 \times 10^{-15}$ |
| 0.020 | 2.6 | $1.10 \times 10^{-6}$ | $6.24 \times 10^{-16}$ |
| 0.010 | 1.3 | $4.01 \times 10^{-5}$ | $2.93 \times 10^{-16}$ |

#### Key Findings

1. **Pseudo-spectral (DST) achieves machine precision**: For all Gaussian widths tested, the DST-based solver achieves $\sim 10^{-15}$ to $10^{-16}$ error regardless of $\sigma$.

2. **Real-space has spatial discretization error**: Error increases when $\sigma/\Delta x < 3$ (under-resolved Gaussian). For well-resolved cases ($\sigma/\Delta x > 5$), error is $\sim 10^{-6}$.

3. **Real-space convergence order**: Approximately $p \approx 2$ for CN-ADI2, reaching a spatial error floor at fine $\Delta s$.

4. **Absorbing BCs work correctly**: Both methods properly handle Dirichlet boundary conditions, with propagators decaying to $\sim 10^{-15}$ at boundaries.

5. **Resolution requirement**: Real-space method needs $\sigma/\Delta x \gtrsim 3$-$5$ for accurate results with sharp initial conditions.

#### Method Selection for Absorbing Boundaries

| Criterion | Recommended Method |
|-----------|-------------------|
| Maximum accuracy | Pseudo-spectral (DST) |
| Sharp initial condition ($\sigma/\Delta x < 3$) | Pseudo-spectral (DST) |
| Simple geometry, periodic in other directions | Pseudo-spectral (DST) |
| Non-uniform grids or complex geometries | Real-space (with sufficient resolution) |

#### Stability Warning: Grafting Points Near Boundaries

**Important**: CN-ADI4 (with Richardson extrapolation) can become unstable when the initial condition (grafting point) is close to an absorbing boundary. Testing shows:

| $x_0/\sigma$ from boundary | CN-ADI4 | CN-ADI2 |
|-------------------|---------|---------|
| $> 5\sigma$ | Stable | Stable |
| $2$-$3\sigma$ | **Unstable** | Stable |
| $< 2\sigma$ | **Diverges** | Stable |

**Recommendation**: For grafted brush simulations:
- Use pseudo-spectral (DST) when possible - it achieves spectral accuracy
- For real-space: ensure $\sigma/\Delta x > 5$ for accuracy, and use CN-ADI2 (avoid CN-ADI4) near boundaries

## Usage

### Python Interface

```python
params = {
    "nx": [32, 32, 32],
    "lx": [4.0, 4.0, 4.0],
    "chain_model": "continuous",
    "ds": 0.01,

    # Boundary conditions: "periodic", "reflecting", or "absorbing"
    "bc": ["reflecting", "reflecting",    # x-direction (low, high)
           "reflecting", "reflecting",    # y-direction (low, high)
           "absorbing", "absorbing"],     # z-direction (low, high)

    # Select real-space method: "cn-adi2" or "cn-adi4"
    "numerical_method": "cn-adi2",

    # ... other parameters
}
```

### Mixed Boundary Conditions

Different boundary conditions can be specified for each direction:

```python
# Confined film: reflecting in z, periodic in x and y
"bc": ["periodic", "periodic",      # x: periodic
       "periodic", "periodic",      # y: periodic
       "reflecting", "reflecting"]  # z: reflecting (confined)

# Semi-infinite slab: absorbing on one side
"bc": ["periodic", "periodic",
       "periodic", "periodic",
       "reflecting", "absorbing"]   # z: reflecting bottom, absorbing top
```

### Grafted Brush Example

For polymer brushes grafted to a surface, use a delta-function initial condition with absorbing boundaries:

```python
from polymerfts import PropagatorSolver
import numpy as np

# 1D grafted brush with absorbing boundaries
solver = PropagatorSolver(
    nx=[128],
    lx=[4.0],
    ds=0.01,
    bond_lengths={"A": 1.0},
    bc=["absorbing", "absorbing"],  # Absorbing on both sides
    chain_model="continuous",
    numerical_method="cn-adi2",     # or "cn-adi4" for 4th-order accuracy
    platform="cpu-mkl",
    reduce_memory_usage=False
)

# Add polymer with grafting point at node 0
solver.add_polymer(
    volume_fraction=1.0,
    blocks=[["A", 1.0, 0, 1]],
    grafting_points={0: "G"}  # Node 0 uses custom q_init
)

# Create delta-function initial condition at grafting point
dx = 4.0 / 128
x = (np.arange(128) + 0.5) * dx
x0 = 0.5  # Grafting point position

# Gaussian approximation to delta function
sigma = 0.1
q_init = np.exp(-(x - x0)**2 / (2 * sigma**2))
q_init = q_init / (np.sum(q_init) * dx)  # Normalize

# Compute propagators with zero field
w_field = np.zeros(128)
solver.compute_propagators({"A": w_field}, q_init={"G": q_init})

# Get propagator at chain end
q_end = solver.get_propagator(polymer=0, v=0, u=1, step=100)
Q = solver.get_partition_function(polymer=0)
```

## Implementation Details

### Files

| File | Description |
|------|-------------|
| `src/platforms/cpu/CpuSolverCNADI.cpp` | CPU CN-ADI2/CN-ADI4 solver |
| `src/platforms/cpu/CpuSolverCNADI.h` | CPU header |
| `src/platforms/cuda/CudaSolverCNADI.cu` | CUDA CN-ADI2/CN-ADI4 solver |
| `src/platforms/cuda/CudaSolverCNADI.h` | CUDA header |
| `src/platforms/cpu/CpuComputationGlobalRichardson.cpp` | CPU CN-ADI4-GQ computation |
| `src/platforms/cpu/CpuComputationGlobalRichardson.h` | CPU header |
| `src/platforms/cpu/CpuSolverGlobalRichardsonBase.cpp` | CPU CN-ADI4-GQ base solver |
| `src/platforms/cpu/CpuSolverGlobalRichardsonBase.h` | CPU header |
| `src/platforms/cuda/CudaComputationGlobalRichardson.cu` | CUDA CN-ADI4-GQ computation |
| `src/platforms/cuda/CudaComputationGlobalRichardson.h` | CUDA header |
| `src/platforms/cuda/CudaSolverGlobalRichardsonBase.cu` | CUDA CN-ADI4-GQ base solver |
| `src/platforms/cuda/CudaSolverGlobalRichardsonBase.h` | CUDA header |
| `src/common/FiniteDifference.cpp` | Tridiagonal coefficient generation |

### Tridiagonal Solvers

**Non-periodic (Thomas Algorithm)**:
- Forward elimination followed by back substitution
- CUDA: Uses shared memory for coefficient caching
- CPU: Direct sequential solve

**Periodic (Sherman-Morrison)**:
- Converts cyclic system to standard tridiagonal + correction
- Solves two systems and combines results
- CUDA: Optimized with register reuse

### CUDA Optimizations

- **Shared memory**: Tridiagonal coefficients cached in shared memory
- **Coalesced access**: Data layout optimized for memory coalescing
- **Stream support**: Multiple propagators computed concurrently
- **Dynamic parallelism**: Each thread handles one tridiagonal system

## Limitations

1. **Stress computation**: Not yet implemented for real-space method
2. **Discrete chains**: Only continuous chain model supported
3. **Non-orthogonal cells**: Only orthogonal unit cells supported
4. **Complex fields**: Only real-valued fields supported

## When to Use Real-Space vs Pseudo-Spectral

### Use Real-Space When:
- Non-periodic boundary conditions are required (confined systems, interfaces)
- Small grid sizes where real-space is competitive
- CN-ADI4 (4th-order) accuracy is needed with non-periodic boundaries

### Use Pseudo-Spectral When:
- Periodic boundary conditions are acceptable
- Large grid sizes (pseudo-spectral scales better)
- Stress calculations are needed
- Maximum performance is required

## References

1. **Crank-Nicolson Method**: J. Crank and P. Nicolson, "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type", *Proc. Cambridge Phil. Soc.*, **1947**, 43, 50-67.

2. **ADI Method**: D. W. Peaceman and H. H. Rachford, "The numerical solution of parabolic and elliptic differential equations", *J. Soc. Indust. Appl. Math.*, **1955**, 3, 28-41.

3. **Richardson Extrapolation**: L. F. Richardson, "The approximate arithmetical solution by finite differences of physical problems involving differential equations", *Phil. Trans. R. Soc. A*, **1911**, 210, 307-357.

4. **Thomas Algorithm**: L. H. Thomas, "Elliptic problems in linear differential equations over a network", Watson Sci. Comput. Lab Report, Columbia University, **1949**.
