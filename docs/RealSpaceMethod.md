# Real-Space Method Documentation

This document describes the real-space finite difference methods for continuous chain propagators, including CN-ADI and SDC solvers, boundary condition support, and usage.

For performance benchmarks and comparison with pseudo-spectral methods, see [NumericalMethodsPerformance.md](NumericalMethodsPerformance.md).

## Overview

The real-space methods use finite difference discretization to solve the modified diffusion equation:

$$\frac{\partial q}{\partial s} = \frac{b^2}{6} \nabla^2 q - w(\mathbf{r}) q$$

Both pseudo-spectral and real-space methods support the following boundary conditions:

| Boundary Condition | Description | Mathematical Form |
|-------------------|-------------|-------------------|
| **Periodic** | Cyclic boundary | $q(0) = q(L)$ |
| **Reflecting** | Neumann (zero flux) | $\partial q / \partial n = 0$ |
| **Absorbing** | Dirichlet (zero value) | $q = 0$ at boundary |

## Grid Discretization

The real-space methods use a **cell-centered grid** where grid points are located at cell centers:

$$x_i = \left(i + \frac{1}{2}\right) \Delta x, \quad i = 0, 1, \ldots, N-1$$

Boundaries are at cell faces ($x = 0$ and $x = L$), not at grid points.

**Boundary condition implementation using ghost cells:**

| BC Type | Ghost Cell Value | Effect on Laplacian |
|---------|------------------|---------------------|
| **Periodic** | $q_{-1} = q_{N-1}$ | Cyclic tridiagonal system |
| **Reflecting** | $q_{-1} = q_0$ (symmetric) | Diagonal += off-diagonal |
| **Absorbing** | $q_{-1} = -q_0$ (antisymmetric) | Diagonal -= off-diagonal |

The antisymmetric ghost for absorbing BC enforces $q = 0$ at the cell face (halfway between ghost and first cell).

## Numerical Methods

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

## Available Methods

| Method | Order | Description |
|--------|-------|-------------|
| **cn-adi2** | 2nd | Standard Crank-Nicolson ADI |
| **cn-adi4-lr** | 4th | CN-ADI with Local Richardson extrapolation |
| **cn-adi4-gr** | 4th | CN-ADI with Global Richardson extrapolation |
| **sdc-N** | Nth | Spectral Deferred Correction (N=2-10) |

SDC uses PCG solver (no ADI splitting error). Specify order as `sdc-3`, `sdc-5`, `sdc-7`, etc.

### Richardson Extrapolation for 4th-Order Accuracy

The 4th-order methods use Richardson extrapolation combining one full step with two half-steps:

$$q_{\text{out}} = \frac{4 \cdot q_{\text{half}} - q_{\text{full}}}{3}$$

This cancels the $O(\Delta s^2)$ error term, yielding $O(\Delta s^4)$ accuracy.

### CN-ADI4-LR vs CN-ADI4-GR

Both methods achieve 4th-order convergence but differ in where Richardson extrapolation is applied:

**CN-ADI4-LR (Local Richardson)** - Extrapolation at each propagator step:
```python
for n in range(N_steps):
    q_full[n+1] = advance(q[n], ds)
    q_half_temp = advance(q[n], ds/2)
    q_half[n+1] = advance(q_half_temp, ds/2)
    q[n+1] = (4*q_half[n+1] - q_full[n+1]) / 3
```

**CN-ADI4-GR (Global Richardson)** - Extrapolation at quadrature level:
```python
# Advance two independent chains with different step sizes
for n in range(N_steps):
    q_full[n+1] = advance(q_full[n], ds)

for n in range(2*N_steps):
    q_half[n+1] = advance(q_half[n], ds/2)

# Richardson extrapolation applied to final quantities
Q_rich = (4*Q_half - Q_full) / 3
phi_rich = (4*phi_half - phi_full) / 3
```

| Factor | CN-ADI4-LR | CN-ADI4-GR |
|--------|------------|------------|
| Convergence (1D) | ~3.5 order | True 4th order |
| Convergence (3D) | ~3.7 order | ~4th order (limited by spatial error) |
| Memory | Lower | Higher (stores two chains) |
| ADI steps per N | 3N | 3N |

**Recommendation**: Both methods have similar computational cost. CN-ADI4-GR achieves true 4th-order convergence but requires more memory. For most applications, CN-ADI4-LR is sufficient.

### Spectral Deferred Correction (SDC)

SDC is an iterative method that uses spectral quadrature (Gauss-Lobatto nodes) to achieve high-order accuracy. The implementation uses a **fully implicit scheme** where both diffusion and reaction terms are included in the implicit matrix, ensuring material conservation.

**Fully Implicit Formulation:**

The implicit matrix includes both diffusion and reaction terms:

$$(I - \Delta\tau \cdot D\nabla^2 + \Delta\tau \cdot w) q^{n+1} = q^n$$

This is in contrast to IMEX splitting which would apply the reaction term explicitly:

$$(I - \Delta\tau \cdot D\nabla^2) q^{n+1} = e^{-w\Delta\tau} q^n \quad \text{(IMEX - not used)}$$

The fully implicit scheme ensures **material conservation**: the forward and backward partition functions are equal to machine precision (~10⁻¹⁵), which is required for accurate SCFT calculations.

**Algorithm per contour step:**

1. **Discretize** the interval $[s_n, s_{n+1}]$ using $M$ Gauss-Lobatto nodes $\tau_m \in [0, 1]$:
   - For $M=3$: $\tau = [0, 0.5, 1]$
   - For $M=4$: $\tau = [0, 0.276..., 0.724..., 1]$

2. **Predictor** (Backward Euler at each sub-interval):
   - Solve implicitly: $(I - \Delta\tau \cdot D\nabla^2 + \Delta\tau \cdot w) X^{[0]}_{m+1} = X^{[0]}_m$
   - 1D: Tridiagonal solver (Thomas algorithm) with $w$ in diagonal
   - 2D/3D: **Preconditioned Conjugate Gradient (PCG)** with Jacobi preconditioner

3. **Corrector** ($K$ iterations):
   - Compute $F_m = D\nabla^2 q_m - w \cdot q_m$ at all nodes
   - Apply spectral integral using Lagrange interpolation:
     $$X^{[k+1]}_{m+1} = X^{[k]}_m + \int_{\tau_m}^{\tau_{m+1}} F^{[k]}(\tau') d\tau' - \Delta\tau \cdot F^{[k]}_{m+1}$$
   - Solve implicit system at each node

**Order of Accuracy:**

The SDC order is $\min(K+1, 2M-2)$ where $M$ is the number of Gauss-Lobatto nodes and $K$ is the number of correction iterations. Unlike CN-ADI methods which use ADI splitting (introducing $O(\Delta s^2)$ error in 2D/3D), SDC uses PCG to solve the full implicit system directly, avoiding splitting errors.

| Method | M | K | Order |
|--------|---|---|-------|
| `sdc-2` | 3 | 1 | 2nd |
| `sdc-3` | 3 | 2 | 3rd |
| `sdc-5` | 4 | 4 | 5th |
| `sdc-7` | 5 | 6 | 7th |

**Material Conservation:**

The fully implicit scheme satisfies the Hermiticity condition $(VU)^\dagger = VU$ where $V$ is the volume matrix and $U$ is the evolution operator. This ensures:
- Forward partition function = Backward partition function
- Partition function differences ~10⁻¹⁵ (machine precision)

### Runtime Selection

```python
from polymerfts import PropagatorSolver

# CN-ADI2 (default - faster)
solver = PropagatorSolver(..., numerical_method="cn-adi2")

# CN-ADI4 with local Richardson (good balance)
solver = PropagatorSolver(..., numerical_method="cn-adi4-lr")

# CN-ADI4 with global Richardson (highest accuracy in 1D)
solver = PropagatorSolver(..., numerical_method="cn-adi4-gr")

# SDC (Spectral Deferred Correction) - specify order as sdc-N (N=2-10)
solver = PropagatorSolver(..., numerical_method="sdc-5")  # 5th order
```

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

    # Select real-space method
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

## Implementation Details

### Files

| File | Description |
|------|-------------|
| `src/common/FiniteDifference.cpp` | Tridiagonal matrix coefficient generation |
| `src/platforms/cpu/CpuSolverCNADI.cpp` | CPU CN-ADI2/CN-ADI4-LR solver |
| `src/platforms/cuda/CudaSolverCNADI.cu` | CUDA CN-ADI2/CN-ADI4-LR solver |
| `src/platforms/cpu/CpuSolverGlobalRichardsonBase.cpp` | CPU CN-ADI4-GR base solver |
| `src/platforms/cuda/CudaSolverGlobalRichardsonBase.cu` | CUDA CN-ADI4-GR base solver |
| `src/platforms/cpu/CpuSolverSDC.cpp` | CPU SDC solver |
| `src/platforms/cuda/CudaSolverSDC.cu` | CUDA SDC solver |

### Tridiagonal Systems

Each ADI sweep requires solving a tridiagonal system $A\mathbf{x} = \mathbf{b}$:

$$\begin{pmatrix} d_0 & h_0 & & & \\ l_1 & d_1 & h_1 & & \\ & \ddots & \ddots & \ddots & \\ & & l_{n-1} & d_{n-1} & h_{n-1} \\ & & & l_n & d_n \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \\ \vdots \\ x_{n-1} \\ x_n \end{pmatrix} = \begin{pmatrix} b_0 \\ b_1 \\ \vdots \\ b_{n-1} \\ b_n \end{pmatrix}$$

**Non-periodic (Thomas Algorithm)**:
- Forward elimination: Modify coefficients to eliminate lower diagonal $l_i$
- Back substitution: Solve for $x_i$ from $i = n$ down to $0$
- Complexity: $O(N)$
- CUDA: Parallel solves with shared memory caching for coefficients

**Periodic (Sherman-Morrison)**:
- Cyclic system has corner elements: $A_{0,n} = h_n$ and $A_{n,0} = l_0$
- Decomposes as $A = B + \mathbf{u}\mathbf{v}^T$ where $B$ is standard tridiagonal
- Solves $B\mathbf{y} = \mathbf{b}$ and $B\mathbf{z} = \mathbf{u}$, then $\mathbf{x} = \mathbf{y} - \frac{\mathbf{v}^T \mathbf{y}}{1 + \mathbf{v}^T \mathbf{z}} \mathbf{z}$
- Complexity: $O(N)$ with constant factor overhead

### ADI Splitting

For 3D problems with grid $(N_x, N_y, N_z)$:

| Sweep | Systems to Solve | System Size |
|-------|------------------|-------------|
| X-sweep | $N_y \times N_z$ | $N_x$ |
| Y-sweep | $N_x \times N_z$ | $N_y$ |
| Z-sweep | $N_x \times N_y$ | $N_z$ |

CUDA parallelizes across systems, solving thousands of tridiagonal systems simultaneously.

## Limitations

1. **Stress computation**: Not yet implemented for real-space method
2. **Discrete chains**: Only continuous chain model supported
3. **Non-orthogonal cells**: Only orthogonal unit cells supported
4. **Complex fields**: Only real-valued fields supported

## When to Use Real-Space vs Pseudo-Spectral

| Criterion | Pseudo-Spectral (RQM4/ETDRK4) | Real-Space (CN-ADI/SDC) |
|-----------|------------------------------|-------------------------|
| **Boundary conditions** | Periodic, reflecting, absorbing | Periodic, reflecting, absorbing |
| **Performance** | Fastest (~3x faster) | Slower |
| **Stress calculation** | Supported | Not yet implemented |
| **Spatial accuracy** | Spectral (exponential) | O(Δx²) |

**Note:** Both pseudo-spectral and real-space methods now support all boundary condition types. Pseudo-spectral methods use DCT (reflecting) and DST (absorbing) transforms, while real-space methods use modified tridiagonal matrices.

### Choosing Between Real-Space Methods

| Method | Best For |
|--------|----------|
| **cn-adi2** | Fast prototyping, coarse grids |
| **cn-adi4-lr** | General use, good accuracy/speed tradeoff |
| **cn-adi4-gr** | High accuracy with memory overhead |
| **sdc-N** | Highest accuracy (no ADI splitting error) |

### Use Pseudo-Spectral (RQM4/ETDRK4) When:
- Stress calculations are needed for box optimization
- Maximum performance is required
- Spectral accuracy is desired

### Use Real-Space (CN-ADI) When:
- Stress calculations are not needed
- Comparing with finite difference implementations

## References

1. **Crank-Nicolson Method**: J. Crank and P. Nicolson, *Proc. Cambridge Phil. Soc.*, **1947**, 43, 50-67.

2. **ADI Method**: D. W. Peaceman and H. H. Rachford, *J. Soc. Indust. Appl. Math.*, **1955**, 3, 28-41.

3. **Richardson Extrapolation**: L. F. Richardson, *Phil. Trans. R. Soc. A*, **1911**, 210, 307-357.

4. **SDC Method**: A. Dutt, L. Greengard, and V. Rokhlin, *BIT Numerical Mathematics*, **2000**, 40, 241-266.

5. **Semi-implicit SDC**: M. L. Minion, *Commun. Math. Sci.*, **2003**, 1, 471-500.
